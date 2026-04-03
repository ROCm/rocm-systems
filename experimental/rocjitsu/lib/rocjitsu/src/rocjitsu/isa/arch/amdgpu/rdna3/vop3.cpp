// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/rdna3/vop3.h"
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

VNopVop3::VNopVop3(const MachineInst *inst)
    : Vop3("v_nop", reinterpret_cast<const OpEncoding *>(inst)) {}

void VNopVop3::execute(amdgpu::Wavefront &wf) { (void)wf; }

VMovB32Vop3::VMovB32Vop3(const MachineInst *inst)
    : Vop3("v_mov_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VMovB32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = std::bit_cast<float>(src0.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    if (inst_.omod == 1)
      s *= 2.0f;
    else if (inst_.omod == 2)
      s *= 4.0f;
    else if (inst_.omod == 3)
      s *= 0.5f;
    if (inst_.clamp)
      s = std::clamp(s, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(s));
  }
}

VReadfirstlaneB32Vop3::VReadfirstlaneB32Vop3(const MachineInst *inst)
    : Vop3("v_readfirstlane_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VReadfirstlaneB32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint32_t val = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (exec & (1ULL << lane)) {
      val = src0.read_lane(wf, lane);
      break;
    }
  }
  vdst.write_scalar(wf, val);
}

VCvtI32F64Vop3::VCvtI32F64Vop3(const MachineInst *inst)
    : Vop3("v_cvt_i32_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtI32F64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s = std::bit_cast<double>(src0.read_lane64(wf, lane));
    int32_t r;
    if (std::isnan(s))
      r = 0;
    else if (s >= 2147483648.0)
      r = INT32_MAX;
    else if (s < -2147483648.0)
      r = INT32_MIN;
    else
      r = static_cast<int32_t>(s);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(r));
  }
}

VCvtF64I32Vop3::VCvtF64I32Vop3(const MachineInst *inst)
    : Vop3("v_cvt_f64_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtF64I32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s = static_cast<int32_t>(src0.read_lane(wf, lane));
    vdst.write_lane64(wf, lane, std::bit_cast<uint64_t>(static_cast<double>(s)));
  }
}

VCvtF32I32Vop3::VCvtF32I32Vop3(const MachineInst *inst)
    : Vop3("v_cvt_f32_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtF32I32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s = static_cast<int32_t>(src0.read_lane(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(static_cast<float>(s)));
  }
}

VCvtF32U32Vop3::VCvtF32U32Vop3(const MachineInst *inst)
    : Vop3("v_cvt_f32_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtF32U32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s = src0.read_lane(wf, lane);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(static_cast<float>(s)));
  }
}

VCvtU32F32Vop3::VCvtU32F32Vop3(const MachineInst *inst)
    : Vop3("v_cvt_u32_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtU32F32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = std::bit_cast<float>(src0.read_lane(wf, lane));
    uint32_t r;
    if (std::isnan(s) || s < 0.0f)
      r = 0;
    else if (s >= 4294967296.0f)
      r = UINT32_MAX;
    else
      r = static_cast<uint32_t>(s);
    vdst.write_lane(wf, lane, r);
  }
}

VCvtI32F32Vop3::VCvtI32F32Vop3(const MachineInst *inst)
    : Vop3("v_cvt_i32_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtI32F32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = std::bit_cast<float>(src0.read_lane(wf, lane));
    int32_t r;
    if (std::isnan(s))
      r = 0;
    else if (s >= 2147483648.0f)
      r = INT32_MAX;
    else if (s < -2147483648.0f)
      r = INT32_MIN;
    else
      r = static_cast<int32_t>(s);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(r));
  }
}

VCvtF16F32Vop3::VCvtF16F32Vop3(const MachineInst *inst)
    : Vop3("v_cvt_f16_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtF16F32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = std::bit_cast<float>(src0.read_lane(wf, lane));
    vdst.write_lane(wf, lane, util::f32_to_f16(s));
  }
}

VCvtF32F16Vop3::VCvtF32F16Vop3(const MachineInst *inst)
    : Vop3("v_cvt_f32_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtF32F16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t raw = src0.read_lane(wf, lane);
    vdst.write_lane(wf, lane,
                    std::bit_cast<uint32_t>(util::f16_to_f32(static_cast<uint16_t>(raw))));
  }
}

VCvtNearestI32F32Vop3::VCvtNearestI32F32Vop3(const MachineInst *inst)
    : Vop3("v_cvt_nearest_i32_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtNearestI32F32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = std::bit_cast<float>(src0.read_lane(wf, lane));
    int32_t r;
    if (std::isnan(s))
      r = 0;
    else if (s >= 2147483648.0f)
      r = INT32_MAX;
    else if (s < -2147483648.0f)
      r = INT32_MIN;
    else
      r = static_cast<int32_t>(s);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(r));
  }
}

VCvtFloorI32F32Vop3::VCvtFloorI32F32Vop3(const MachineInst *inst)
    : Vop3("v_cvt_floor_i32_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtFloorI32F32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = std::bit_cast<float>(src0.read_lane(wf, lane));
    int32_t r;
    if (std::isnan(s))
      r = 0;
    else if (s >= 2147483648.0f)
      r = INT32_MAX;
    else if (s < -2147483648.0f)
      r = INT32_MIN;
    else
      r = static_cast<int32_t>(s);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(r));
  }
}

VCvtOffF32I4Vop3::VCvtOffF32I4Vop3(const MachineInst *inst)
    : Vop3("v_cvt_off_f32_i4", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtOffF32I4Vop3::execute(amdgpu::Wavefront &wf) { (void)wf; }

VCvtF32F64Vop3::VCvtF32F64Vop3(const MachineInst *inst)
    : Vop3("v_cvt_f32_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtF32F64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s = std::bit_cast<double>(src0.read_lane64(wf, lane));
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(static_cast<float>(s)));
  }
}

VCvtF64F32Vop3::VCvtF64F32Vop3(const MachineInst *inst)
    : Vop3("v_cvt_f64_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtF64F32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = std::bit_cast<float>(src0.read_lane(wf, lane));
    vdst.write_lane64(wf, lane, std::bit_cast<uint64_t>(static_cast<double>(s)));
  }
}

VCvtF32Ubyte0Vop3::VCvtF32Ubyte0Vop3(const MachineInst *inst)
    : Vop3("v_cvt_f32_ubyte0", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtF32Ubyte0Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s = src0.read_lane(wf, lane);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(static_cast<float>(s & 0xFFu)));
  }
}

VCvtF32Ubyte1Vop3::VCvtF32Ubyte1Vop3(const MachineInst *inst)
    : Vop3("v_cvt_f32_ubyte1", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtF32Ubyte1Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s = src0.read_lane(wf, lane);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(static_cast<float>((s >> 8) & 0xFFu)));
  }
}

VCvtF32Ubyte2Vop3::VCvtF32Ubyte2Vop3(const MachineInst *inst)
    : Vop3("v_cvt_f32_ubyte2", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtF32Ubyte2Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s = src0.read_lane(wf, lane);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(static_cast<float>((s >> 16) & 0xFFu)));
  }
}

VCvtF32Ubyte3Vop3::VCvtF32Ubyte3Vop3(const MachineInst *inst)
    : Vop3("v_cvt_f32_ubyte3", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtF32Ubyte3Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s = src0.read_lane(wf, lane);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(static_cast<float>((s >> 24) & 0xFFu)));
  }
}

VCvtU32F64Vop3::VCvtU32F64Vop3(const MachineInst *inst)
    : Vop3("v_cvt_u32_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtU32F64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s = std::bit_cast<double>(src0.read_lane64(wf, lane));
    uint32_t r;
    if (std::isnan(s) || s < 0.0)
      r = 0;
    else if (s >= 4294967296.0)
      r = UINT32_MAX;
    else
      r = static_cast<uint32_t>(s);
    vdst.write_lane(wf, lane, r);
  }
}

VCvtF64U32Vop3::VCvtF64U32Vop3(const MachineInst *inst)
    : Vop3("v_cvt_f64_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtF64U32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s = src0.read_lane(wf, lane);
    vdst.write_lane64(wf, lane, std::bit_cast<uint64_t>(static_cast<double>(s)));
  }
}

VTruncF64Vop3::VTruncF64Vop3(const MachineInst *inst)
    : Vop3("v_trunc_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VTruncF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s = std::bit_cast<double>(src0.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    double result = std::trunc(s);
    if (inst_.omod == 1)
      result *= 2.0;
    else if (inst_.omod == 2)
      result *= 4.0;
    else if (inst_.omod == 3)
      result *= 0.5;
    if (inst_.clamp)
      result = std::clamp(result, 0.0, 1.0);
    vdst.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));
  }
}

VCeilF64Vop3::VCeilF64Vop3(const MachineInst *inst)
    : Vop3("v_ceil_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCeilF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s = std::bit_cast<double>(src0.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    double result = std::ceil(s);
    if (inst_.omod == 1)
      result *= 2.0;
    else if (inst_.omod == 2)
      result *= 4.0;
    else if (inst_.omod == 3)
      result *= 0.5;
    if (inst_.clamp)
      result = std::clamp(result, 0.0, 1.0);
    vdst.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));
  }
}

VRndneF64Vop3::VRndneF64Vop3(const MachineInst *inst)
    : Vop3("v_rndne_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VRndneF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s = std::bit_cast<double>(src0.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    double result = std::nearbyint(s);
    if (inst_.omod == 1)
      result *= 2.0;
    else if (inst_.omod == 2)
      result *= 4.0;
    else if (inst_.omod == 3)
      result *= 0.5;
    if (inst_.clamp)
      result = std::clamp(result, 0.0, 1.0);
    vdst.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));
  }
}

VFloorF64Vop3::VFloorF64Vop3(const MachineInst *inst)
    : Vop3("v_floor_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VFloorF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s = std::bit_cast<double>(src0.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    double result = std::floor(s);
    if (inst_.omod == 1)
      result *= 2.0;
    else if (inst_.omod == 2)
      result *= 4.0;
    else if (inst_.omod == 3)
      result *= 0.5;
    if (inst_.clamp)
      result = std::clamp(result, 0.0, 1.0);
    vdst.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));
  }
}

VPipeflushVop3::VPipeflushVop3(const MachineInst *inst)
    : Vop3("v_pipeflush", reinterpret_cast<const OpEncoding *>(inst)) {}

void VPipeflushVop3::execute(amdgpu::Wavefront &wf) { (void)wf; }

VMovB16Vop3::VMovB16Vop3(const MachineInst *inst)
    : Vop3("v_mov_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VMovB16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = std::bit_cast<float>(src0.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    if (inst_.omod == 1)
      s *= 2.0f;
    else if (inst_.omod == 2)
      s *= 4.0f;
    else if (inst_.omod == 3)
      s *= 0.5f;
    if (inst_.clamp)
      s = std::clamp(s, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(s));
  }
}

VFractF32Vop3::VFractF32Vop3(const MachineInst *inst)
    : Vop3("v_fract_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VFractF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = std::bit_cast<float>(src0.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    float result = s - std::floor(s);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VTruncF32Vop3::VTruncF32Vop3(const MachineInst *inst)
    : Vop3("v_trunc_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VTruncF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = std::bit_cast<float>(src0.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    float result = std::trunc(s);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VCeilF32Vop3::VCeilF32Vop3(const MachineInst *inst)
    : Vop3("v_ceil_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCeilF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = std::bit_cast<float>(src0.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    float result = std::ceil(s);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VRndneF32Vop3::VRndneF32Vop3(const MachineInst *inst)
    : Vop3("v_rndne_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VRndneF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = std::bit_cast<float>(src0.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    float result = std::nearbyint(s);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VFloorF32Vop3::VFloorF32Vop3(const MachineInst *inst)
    : Vop3("v_floor_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VFloorF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = std::bit_cast<float>(src0.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    float result = std::floor(s);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VExpF32Vop3::VExpF32Vop3(const MachineInst *inst)
    : Vop3("v_exp_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VExpF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = std::bit_cast<float>(src0.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    float result = amdgpu::transcendental::exp_f32(s);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VLogF32Vop3::VLogF32Vop3(const MachineInst *inst)
    : Vop3("v_log_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VLogF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = std::bit_cast<float>(src0.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    float result = amdgpu::transcendental::log_f32(s);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VRcpF32Vop3::VRcpF32Vop3(const MachineInst *inst)
    : Vop3("v_rcp_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VRcpF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = std::bit_cast<float>(src0.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    float result = amdgpu::transcendental::rcp_f32(s);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VRcpIflagF32Vop3::VRcpIflagF32Vop3(const MachineInst *inst)
    : Vop3("v_rcp_iflag_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VRcpIflagF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = std::bit_cast<float>(src0.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    float result = amdgpu::transcendental::rcp_f32(s);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VRsqF32Vop3::VRsqF32Vop3(const MachineInst *inst)
    : Vop3("v_rsq_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VRsqF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = std::bit_cast<float>(src0.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    float result = amdgpu::transcendental::rsq_f32(s);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VRcpF64Vop3::VRcpF64Vop3(const MachineInst *inst)
    : Vop3("v_rcp_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VRcpF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s = std::bit_cast<double>(src0.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    double result = amdgpu::transcendental::rcp_f64(s);
    if (inst_.omod == 1)
      result *= 2.0;
    else if (inst_.omod == 2)
      result *= 4.0;
    else if (inst_.omod == 3)
      result *= 0.5;
    if (inst_.clamp)
      result = std::clamp(result, 0.0, 1.0);
    vdst.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));
  }
}

VRsqF64Vop3::VRsqF64Vop3(const MachineInst *inst)
    : Vop3("v_rsq_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VRsqF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s = std::bit_cast<double>(src0.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    double result = amdgpu::transcendental::rsq_f64(s);
    if (inst_.omod == 1)
      result *= 2.0;
    else if (inst_.omod == 2)
      result *= 4.0;
    else if (inst_.omod == 3)
      result *= 0.5;
    if (inst_.clamp)
      result = std::clamp(result, 0.0, 1.0);
    vdst.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));
  }
}

VSqrtF32Vop3::VSqrtF32Vop3(const MachineInst *inst)
    : Vop3("v_sqrt_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VSqrtF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = std::bit_cast<float>(src0.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    float result = amdgpu::transcendental::sqrt_f32(s);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VSqrtF64Vop3::VSqrtF64Vop3(const MachineInst *inst)
    : Vop3("v_sqrt_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VSqrtF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s = std::bit_cast<double>(src0.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    double result = amdgpu::transcendental::sqrt_f64(s);
    if (inst_.omod == 1)
      result *= 2.0;
    else if (inst_.omod == 2)
      result *= 4.0;
    else if (inst_.omod == 3)
      result *= 0.5;
    if (inst_.clamp)
      result = std::clamp(result, 0.0, 1.0);
    vdst.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));
  }
}

VSinF32Vop3::VSinF32Vop3(const MachineInst *inst)
    : Vop3("v_sin_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VSinF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = std::bit_cast<float>(src0.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    float result = amdgpu::transcendental::sin_f32(s);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VCosF32Vop3::VCosF32Vop3(const MachineInst *inst)
    : Vop3("v_cos_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCosF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = std::bit_cast<float>(src0.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    float result = amdgpu::transcendental::cos_f32(s);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VNotB32Vop3::VNotB32Vop3(const MachineInst *inst)
    : Vop3("v_not_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VNotB32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s = src0.read_lane(wf, lane);
    vdst.write_lane(wf, lane, ~s);
  }
}

VBfrevB32Vop3::VBfrevB32Vop3(const MachineInst *inst)
    : Vop3("v_bfrev_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VBfrevB32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s = src0.read_lane(wf, lane);
    uint32_t result = 0;
    for (int i = 0; i < 32; ++i)
      result |= ((s >> i) & 1) << (31 - i);
    vdst.write_lane(wf, lane, result);
  }
}

VClzI32U32Vop3::VClzI32U32Vop3(const MachineInst *inst)
    : Vop3("v_clz_i32_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VClzI32U32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s = src0.read_lane(wf, lane);
    vdst.write_lane(
        wf, lane, s == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countl_zero(s)));
  }
}

VCtzI32B32Vop3::VCtzI32B32Vop3(const MachineInst *inst)
    : Vop3("v_ctz_i32_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCtzI32B32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s = src0.read_lane(wf, lane);
    vdst.write_lane(
        wf, lane, s == 0 ? static_cast<uint32_t>(-1) : static_cast<uint32_t>(std::countr_zero(s)));
  }
}

VClsI32Vop3::VClsI32Vop3(const MachineInst *inst)
    : Vop3("v_cls_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VClsI32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s = src0.read_lane(wf, lane);
    int32_t sv = static_cast<int32_t>(s);
    uint32_t abs_val = sv < 0 ? ~s : s;
    vdst.write_lane(wf, lane,
                    abs_val == 0 ? static_cast<uint32_t>(-1)
                                 : static_cast<uint32_t>(std::countl_zero(abs_val)));
  }
}

VFrexpExpI32F64Vop3::VFrexpExpI32F64Vop3(const MachineInst *inst)
    : Vop3("v_frexp_exp_i32_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VFrexpExpI32F64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s = std::bit_cast<double>(src0.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    int exp = 0;
    if (s != 0.0 && !std::isnan(s) && !std::isinf(s))
      std::frexp(s, &exp);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(exp));
  }
}

VFrexpMantF64Vop3::VFrexpMantF64Vop3(const MachineInst *inst)
    : Vop3("v_frexp_mant_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VFrexpMantF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s = std::bit_cast<double>(src0.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    int exp = 0;
    double result = std::frexp(s, &exp);
    if (inst_.omod == 1)
      result *= 2.0;
    else if (inst_.omod == 2)
      result *= 4.0;
    else if (inst_.omod == 3)
      result *= 0.5;
    if (inst_.clamp)
      result = std::clamp(result, 0.0, 1.0);
    vdst.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));
  }
}

VFractF64Vop3::VFractF64Vop3(const MachineInst *inst)
    : Vop3("v_fract_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VFractF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s = std::bit_cast<double>(src0.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    double result = s - std::floor(s);
    if (inst_.omod == 1)
      result *= 2.0;
    else if (inst_.omod == 2)
      result *= 4.0;
    else if (inst_.omod == 3)
      result *= 0.5;
    if (inst_.clamp)
      result = std::clamp(result, 0.0, 1.0);
    vdst.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));
  }
}

VFrexpExpI32F32Vop3::VFrexpExpI32F32Vop3(const MachineInst *inst)
    : Vop3("v_frexp_exp_i32_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VFrexpExpI32F32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = std::bit_cast<float>(src0.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    int exp = 0;
    if (s != 0.0f && !std::isnan(s) && !std::isinf(s))
      std::frexp(s, &exp);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(exp));
  }
}

VFrexpMantF32Vop3::VFrexpMantF32Vop3(const MachineInst *inst)
    : Vop3("v_frexp_mant_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VFrexpMantF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = std::bit_cast<float>(src0.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    int exp = 0;
    float result = std::frexp(s, &exp);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VMovreldB32Vop3::VMovreldB32Vop3(const MachineInst *inst)
    : Vop3("v_movreld_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VMovreldB32Vop3::execute(amdgpu::Wavefront &wf) { (void)wf; }

VMovrelsB32Vop3::VMovrelsB32Vop3(const MachineInst *inst)
    : Vop3("v_movrels_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VMovrelsB32Vop3::execute(amdgpu::Wavefront &wf) { (void)wf; }

VMovrelsdB32Vop3::VMovrelsdB32Vop3(const MachineInst *inst)
    : Vop3("v_movrelsd_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VMovrelsdB32Vop3::execute(amdgpu::Wavefront &wf) { (void)wf; }

VMovrelsd2B32Vop3::VMovrelsd2B32Vop3(const MachineInst *inst)
    : Vop3("v_movrelsd_2_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VMovrelsd2B32Vop3::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VCvtF16U16Vop3::VCvtF16U16Vop3(const MachineInst *inst)
    : Vop3("v_cvt_f16_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtF16U16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s = static_cast<uint16_t>(src0.read_lane(wf, lane));
    vdst.write_lane(wf, lane, util::f32_to_f16(static_cast<float>(s)));
  }
}

VCvtF16I16Vop3::VCvtF16I16Vop3(const MachineInst *inst)
    : Vop3("v_cvt_f16_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtF16I16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t s = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    vdst.write_lane(wf, lane, util::f32_to_f16(static_cast<float>(s)));
  }
}

VCvtU16F16Vop3::VCvtU16F16Vop3(const MachineInst *inst)
    : Vop3("v_cvt_u16_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtU16F16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    uint16_t r;
    if (std::isnan(s) || s < 0.0f)
      r = 0;
    else if (s >= 65536.0f)
      r = UINT16_MAX;
    else
      r = static_cast<uint16_t>(s);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(r));
  }
}

VCvtI16F16Vop3::VCvtI16F16Vop3(const MachineInst *inst)
    : Vop3("v_cvt_i16_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtI16F16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    int16_t r;
    if (std::isnan(s))
      r = 0;
    else if (s >= 32768.0f)
      r = INT16_MAX;
    else if (s < -32768.0f)
      r = INT16_MIN;
    else
      r = static_cast<int16_t>(s);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(static_cast<uint16_t>(r)));
  }
}

VRcpF16Vop3::VRcpF16Vop3(const MachineInst *inst)
    : Vop3("v_rcp_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VRcpF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    float result = 1.0f / s;
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, util::f32_to_f16(result));
  }
}

VSqrtF16Vop3::VSqrtF16Vop3(const MachineInst *inst)
    : Vop3("v_sqrt_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VSqrtF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    float result = std::sqrt(s);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, util::f32_to_f16(result));
  }
}

VRsqF16Vop3::VRsqF16Vop3(const MachineInst *inst)
    : Vop3("v_rsq_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VRsqF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    float result = 1.0f / std::sqrt(s);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, util::f32_to_f16(result));
  }
}

VLogF16Vop3::VLogF16Vop3(const MachineInst *inst)
    : Vop3("v_log_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VLogF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    float result = std::log2(s);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, util::f32_to_f16(result));
  }
}

VExpF16Vop3::VExpF16Vop3(const MachineInst *inst)
    : Vop3("v_exp_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VExpF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    float result = std::exp2(s);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, util::f32_to_f16(result));
  }
}

VFrexpMantF16Vop3::VFrexpMantF16Vop3(const MachineInst *inst)
    : Vop3("v_frexp_mant_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VFrexpMantF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = std::bit_cast<float>(src0.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    int exp = 0;
    float result = std::frexp(s, &exp);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VFrexpExpI16F16Vop3::VFrexpExpI16F16Vop3(const MachineInst *inst)
    : Vop3("v_frexp_exp_i16_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VFrexpExpI16F16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    int exp = 0;
    if (s != 0.0f && !std::isnan(s) && !std::isinf(s))
      std::frexp(s, &exp);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(exp));
  }
}

VFloorF16Vop3::VFloorF16Vop3(const MachineInst *inst)
    : Vop3("v_floor_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VFloorF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    float result = std::floor(s);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, util::f32_to_f16(result));
  }
}

VCeilF16Vop3::VCeilF16Vop3(const MachineInst *inst)
    : Vop3("v_ceil_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCeilF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    float result = std::ceil(s);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, util::f32_to_f16(result));
  }
}

VTruncF16Vop3::VTruncF16Vop3(const MachineInst *inst)
    : Vop3("v_trunc_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VTruncF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    float result = std::trunc(s);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, util::f32_to_f16(result));
  }
}

VRndneF16Vop3::VRndneF16Vop3(const MachineInst *inst)
    : Vop3("v_rndne_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VRndneF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    float result = std::nearbyint(s);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, util::f32_to_f16(result));
  }
}

VFractF16Vop3::VFractF16Vop3(const MachineInst *inst)
    : Vop3("v_fract_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VFractF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    float result = s - std::floor(s);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, util::f32_to_f16(result));
  }
}

VSinF16Vop3::VSinF16Vop3(const MachineInst *inst)
    : Vop3("v_sin_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VSinF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    float result = std::sin(s * 6.2831853071795864f);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, util::f32_to_f16(result));
  }
}

VCosF16Vop3::VCosF16Vop3(const MachineInst *inst)
    : Vop3("v_cos_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCosF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s = std::fabs(s);
    if (inst_.neg & (1u << 0))
      s = -s;
    float result = std::cos(s * 6.2831853071795864f);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, util::f32_to_f16(result));
  }
}

VSatPkU8I16Vop3::VSatPkU8I16Vop3(const MachineInst *inst)
    : Vop3("v_sat_pk_u8_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VSatPkU8I16Vop3::execute(amdgpu::Wavefront &wf) { (void)wf; }

VCvtNormI16F16Vop3::VCvtNormI16F16Vop3(const MachineInst *inst)
    : Vop3("v_cvt_norm_i16_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtNormI16F16Vop3::execute(amdgpu::Wavefront &wf) { (void)wf; }

VCvtNormU16F16Vop3::VCvtNormU16F16Vop3(const MachineInst *inst)
    : Vop3("v_cvt_norm_u16_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtNormU16F16Vop3::execute(amdgpu::Wavefront &wf) { (void)wf; }

VNotB16Vop3::VNotB16Vop3(const MachineInst *inst)
    : Vop3("v_not_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VNotB16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s = src0.read_lane(wf, lane);
    vdst.write_lane(wf, lane, ~s);
  }
}

VCvtI32I16Vop3::VCvtI32I16Vop3(const MachineInst *inst)
    : Vop3("v_cvt_i32_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtI32I16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s = static_cast<int32_t>(src0.read_lane(wf, lane));
    vdst.write_lane(wf, lane,
                    static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(s & 0xFFFF))));
  }
}

VCvtU32U16Vop3::VCvtU32U16Vop3(const MachineInst *inst)
    : Vop3("v_cvt_u32_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
}

void VCvtU32U16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s = src0.read_lane(wf, lane);
    vdst.write_lane(wf, lane, s & 0xFFFFu);
  }
}

VCndmaskB32Vop3::VCndmaskB32Vop3(const MachineInst *inst)
    : Vop3("v_cndmask_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VCndmaskB32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    float val = (vcc & (1ULL << lane)) ? s1 : s0;
    if (inst_.omod == 1)
      val *= 2.0f;
    else if (inst_.omod == 2)
      val *= 4.0f;
    else if (inst_.omod == 3)
      val *= 0.5f;
    if (inst_.clamp)
      val = std::clamp(val, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(val));
  }
}

VAddF32Vop3::VAddF32Vop3(const MachineInst *inst)
    : Vop3("v_add_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VAddF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float sv1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      sv0 = std::fabs(sv0);
    if (inst_.neg & (1u << 0))
      sv0 = -sv0;
    if (inst_.abs & (1u << 1))
      sv1 = std::fabs(sv1);
    if (inst_.neg & (1u << 1))
      sv1 = -sv1;
    float result = sv0 + sv1;
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VSubF32Vop3::VSubF32Vop3(const MachineInst *inst)
    : Vop3("v_sub_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VSubF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float sv1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      sv0 = std::fabs(sv0);
    if (inst_.neg & (1u << 0))
      sv0 = -sv0;
    if (inst_.abs & (1u << 1))
      sv1 = std::fabs(sv1);
    if (inst_.neg & (1u << 1))
      sv1 = -sv1;
    float result = sv0 - sv1;
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VSubrevF32Vop3::VSubrevF32Vop3(const MachineInst *inst)
    : Vop3("v_subrev_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VSubrevF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float sv1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      sv0 = std::fabs(sv0);
    if (inst_.neg & (1u << 0))
      sv0 = -sv0;
    if (inst_.abs & (1u << 1))
      sv1 = std::fabs(sv1);
    if (inst_.neg & (1u << 1))
      sv1 = -sv1;
    float result = sv1 - sv0;
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VFmacDx9ZeroF32Vop3::VFmacDx9ZeroF32Vop3(const MachineInst *inst)
    : Vop3("v_fmac_dx9_zero_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  src_operands_.emplace_back(&vdst);
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VFmacDx9ZeroF32Vop3::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMulDx9ZeroF32Vop3::VMulDx9ZeroF32Vop3(const MachineInst *inst)
    : Vop3("v_mul_dx9_zero_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMulDx9ZeroF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float sv1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      sv0 = std::fabs(sv0);
    if (inst_.neg & (1u << 0))
      sv0 = -sv0;
    if (inst_.abs & (1u << 1))
      sv1 = std::fabs(sv1);
    if (inst_.neg & (1u << 1))
      sv1 = -sv1;
    float result = sv0 == 0.0f || sv1 == 0.0f ? 0.0f : sv0 * sv1;
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VMulF32Vop3::VMulF32Vop3(const MachineInst *inst)
    : Vop3("v_mul_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMulF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float sv1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      sv0 = std::fabs(sv0);
    if (inst_.neg & (1u << 0))
      sv0 = -sv0;
    if (inst_.abs & (1u << 1))
      sv1 = std::fabs(sv1);
    if (inst_.neg & (1u << 1))
      sv1 = -sv1;
    float result = sv0 * sv1;
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VMulI32I24Vop3::VMulI32I24Vop3(const MachineInst *inst)
    : Vop3("v_mul_i32_i24", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMulI32I24Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t sv0 = static_cast<int32_t>(src0.read_lane(wf, lane) << 8) >> 8;
    int32_t sv1 = static_cast<int32_t>(src1.read_lane(wf, lane) << 8) >> 8;
    vdst.write_lane(wf, lane, static_cast<uint32_t>(sv0 * sv1));
  }
}

VMulHiI32I24Vop3::VMulHiI32I24Vop3(const MachineInst *inst)
    : Vop3("v_mul_hi_i32_i24", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMulHiI32I24Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t sv0 = static_cast<int32_t>(src0.read_lane(wf, lane) << 8) >> 8;
    int32_t sv1 = static_cast<int32_t>(src1.read_lane(wf, lane) << 8) >> 8;
    vdst.write_lane(wf, lane, static_cast<uint32_t>((static_cast<int64_t>(sv0) * sv1) >> 32));
  }
}

VMulU32U24Vop3::VMulU32U24Vop3(const MachineInst *inst)
    : Vop3("v_mul_u32_u24", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMulU32U24Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane) & 0x00FFFFFFu;
    uint32_t sv1 = src1.read_lane(wf, lane) & 0x00FFFFFFu;
    vdst.write_lane(wf, lane, sv0 * sv1);
  }
}

VMulHiU32U24Vop3::VMulHiU32U24Vop3(const MachineInst *inst)
    : Vop3("v_mul_hi_u32_u24", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMulHiU32U24Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane) & 0x00FFFFFFu;
    uint32_t sv1 = src1.read_lane(wf, lane) & 0x00FFFFFFu;
    vdst.write_lane(wf, lane, static_cast<uint32_t>((static_cast<uint64_t>(sv0) * sv1) >> 32));
  }
}

VMinF32Vop3::VMinF32Vop3(const MachineInst *inst)
    : Vop3("v_min_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMinF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float sv1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      sv0 = std::fabs(sv0);
    if (inst_.neg & (1u << 0))
      sv0 = -sv0;
    if (inst_.abs & (1u << 1))
      sv1 = std::fabs(sv1);
    if (inst_.neg & (1u << 1))
      sv1 = -sv1;
    float result = std::fmin(sv0, sv1);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VMaxF32Vop3::VMaxF32Vop3(const MachineInst *inst)
    : Vop3("v_max_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMaxF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float sv1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      sv0 = std::fabs(sv0);
    if (inst_.neg & (1u << 0))
      sv0 = -sv0;
    if (inst_.abs & (1u << 1))
      sv1 = std::fabs(sv1);
    if (inst_.neg & (1u << 1))
      sv1 = -sv1;
    float result = std::fmax(sv0, sv1);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VMinI32Vop3::VMinI32Vop3(const MachineInst *inst)
    : Vop3("v_min_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMinI32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t sv0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t sv1 = static_cast<int32_t>(src1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(sv0 < sv1 ? sv0 : sv1));
  }
}

VMaxI32Vop3::VMaxI32Vop3(const MachineInst *inst)
    : Vop3("v_max_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMaxI32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t sv0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t sv1 = static_cast<int32_t>(src1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(sv0 > sv1 ? sv0 : sv1));
  }
}

VMinU32Vop3::VMinU32Vop3(const MachineInst *inst)
    : Vop3("v_min_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMinU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = src1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, sv0 < sv1 ? sv0 : sv1);
  }
}

VMaxU32Vop3::VMaxU32Vop3(const MachineInst *inst)
    : Vop3("v_max_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMaxU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = src1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, sv0 > sv1 ? sv0 : sv1);
  }
}

VLshlrevB32Vop3::VLshlrevB32Vop3(const MachineInst *inst)
    : Vop3("v_lshlrev_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VLshlrevB32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = src1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, sv1 << (sv0 & 31u));
  }
}

VLshrrevB32Vop3::VLshrrevB32Vop3(const MachineInst *inst)
    : Vop3("v_lshrrev_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VLshrrevB32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = src1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, sv1 >> (sv0 & 31u));
  }
}

VAshrrevI32Vop3::VAshrrevI32Vop3(const MachineInst *inst)
    : Vop3("v_ashrrev_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VAshrrevI32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t sv0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t sv1 = static_cast<int32_t>(src1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(static_cast<int32_t>(sv1) >> (sv0 & 31)));
  }
}

VAndB32Vop3::VAndB32Vop3(const MachineInst *inst)
    : Vop3("v_and_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VAndB32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = src1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, sv0 & sv1);
  }
}

VOrB32Vop3::VOrB32Vop3(const MachineInst *inst)
    : Vop3("v_or_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VOrB32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = src1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, sv0 | sv1);
  }
}

VXorB32Vop3::VXorB32Vop3(const MachineInst *inst)
    : Vop3("v_xor_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VXorB32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = src1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, sv0 ^ sv1);
  }
}

VXnorB32Vop3::VXnorB32Vop3(const MachineInst *inst)
    : Vop3("v_xnor_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VXnorB32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = src1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, ~(sv0 ^ sv1));
  }
}

VAddCoCiU32Vop3SdstEnc::VAddCoCiU32Vop3SdstEnc(const MachineInst *inst)
    : Vop3SdstEnc("v_add_co_ci_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      sdst(64, OperandType::OPR_SREG, 0),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VAddCoCiU32Vop3SdstEnc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = src1.read_lane(wf, lane);
    uint64_t wide = static_cast<uint64_t>(sv0) + static_cast<uint64_t>(sv1);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(wide));
    if (wide > 0xFFFFFFFFULL)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VSubCoCiU32Vop3SdstEnc::VSubCoCiU32Vop3SdstEnc(const MachineInst *inst)
    : Vop3SdstEnc("v_sub_co_ci_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      sdst(64, OperandType::OPR_SREG, 0),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSubCoCiU32Vop3SdstEnc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = src1.read_lane(wf, lane);
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

VSubrevCoCiU32Vop3SdstEnc::VSubrevCoCiU32Vop3SdstEnc(const MachineInst *inst)
    : Vop3SdstEnc("v_subrev_co_ci_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      sdst(64, OperandType::OPR_SREG, 0),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSubrevCoCiU32Vop3SdstEnc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = src1.read_lane(wf, lane);
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

VAddNcU32Vop3::VAddNcU32Vop3(const MachineInst *inst)
    : Vop3("v_add_nc_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VAddNcU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = src1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, sv0 + sv1);
  }
}

VSubNcU32Vop3::VSubNcU32Vop3(const MachineInst *inst)
    : Vop3("v_sub_nc_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VSubNcU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = src1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, sv0 - sv1);
  }
}

VSubrevNcU32Vop3::VSubrevNcU32Vop3(const MachineInst *inst)
    : Vop3("v_subrev_nc_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VSubrevNcU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = src1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, sv1 - sv0);
  }
}

VFmacF32Vop3::VFmacF32Vop3(const MachineInst *inst)
    : Vop3("v_fmac_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  src_operands_.emplace_back(&vdst);
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VFmacF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float sv1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      sv0 = std::fabs(sv0);
    if (inst_.neg & (1u << 0))
      sv0 = -sv0;
    if (inst_.abs & (1u << 1))
      sv1 = std::fabs(sv1);
    if (inst_.neg & (1u << 1))
      sv1 = -sv1;
    float result = sv0 * sv1 + std::bit_cast<float>(vdst.read_lane(wf, lane));
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VCvtPkRtzF16F32Vop3::VCvtPkRtzF16F32Vop3(const MachineInst *inst)
    : Vop3("v_cvt_pk_rtz_f16_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCvtPkRtzF16F32Vop3::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VAddF16Vop3::VAddF16Vop3(const MachineInst *inst)
    : Vop3("v_add_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VAddF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float sv1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      sv0 = std::fabs(sv0);
    if (inst_.neg & (1u << 0))
      sv0 = -sv0;
    if (inst_.abs & (1u << 1))
      sv1 = std::fabs(sv1);
    if (inst_.neg & (1u << 1))
      sv1 = -sv1;
    float result = sv0 + sv1;
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, util::f32_to_f16(result));
  }
}

VSubF16Vop3::VSubF16Vop3(const MachineInst *inst)
    : Vop3("v_sub_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VSubF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float sv1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      sv0 = std::fabs(sv0);
    if (inst_.neg & (1u << 0))
      sv0 = -sv0;
    if (inst_.abs & (1u << 1))
      sv1 = std::fabs(sv1);
    if (inst_.neg & (1u << 1))
      sv1 = -sv1;
    float result = sv0 - sv1;
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, util::f32_to_f16(result));
  }
}

VSubrevF16Vop3::VSubrevF16Vop3(const MachineInst *inst)
    : Vop3("v_subrev_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VSubrevF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float sv1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      sv0 = std::fabs(sv0);
    if (inst_.neg & (1u << 0))
      sv0 = -sv0;
    if (inst_.abs & (1u << 1))
      sv1 = std::fabs(sv1);
    if (inst_.neg & (1u << 1))
      sv1 = -sv1;
    float result = sv1 - sv0;
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, util::f32_to_f16(result));
  }
}

VMulF16Vop3::VMulF16Vop3(const MachineInst *inst)
    : Vop3("v_mul_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMulF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float sv1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      sv0 = std::fabs(sv0);
    if (inst_.neg & (1u << 0))
      sv0 = -sv0;
    if (inst_.abs & (1u << 1))
      sv1 = std::fabs(sv1);
    if (inst_.neg & (1u << 1))
      sv1 = -sv1;
    float result = sv0 * sv1;
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, util::f32_to_f16(result));
  }
}

VFmacF16Vop3::VFmacF16Vop3(const MachineInst *inst)
    : Vop3("v_fmac_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  src_operands_.emplace_back(&vdst);
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VFmacF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float sv1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      sv0 = std::fabs(sv0);
    if (inst_.neg & (1u << 0))
      sv0 = -sv0;
    if (inst_.abs & (1u << 1))
      sv1 = std::fabs(sv1);
    if (inst_.neg & (1u << 1))
      sv1 = -sv1;
    float result = sv0 * sv1 + util::f16_to_f32(static_cast<uint16_t>(vdst.read_lane(wf, lane)));
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, util::f32_to_f16(result));
  }
}

VMaxF16Vop3::VMaxF16Vop3(const MachineInst *inst)
    : Vop3("v_max_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMaxF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float sv1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      sv0 = std::fabs(sv0);
    if (inst_.neg & (1u << 0))
      sv0 = -sv0;
    if (inst_.abs & (1u << 1))
      sv1 = std::fabs(sv1);
    if (inst_.neg & (1u << 1))
      sv1 = -sv1;
    float result = std::fmax(sv0, sv1);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, util::f32_to_f16(result));
  }
}

VMinF16Vop3::VMinF16Vop3(const MachineInst *inst)
    : Vop3("v_min_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMinF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float sv1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      sv0 = std::fabs(sv0);
    if (inst_.neg & (1u << 0))
      sv0 = -sv0;
    if (inst_.abs & (1u << 1))
      sv1 = std::fabs(sv1);
    if (inst_.neg & (1u << 1))
      sv1 = -sv1;
    float result = std::fmin(sv0, sv1);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, util::f32_to_f16(result));
  }
}

VLdexpF16Vop3::VLdexpF16Vop3(const MachineInst *inst)
    : Vop3("v_ldexp_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VLdexpF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    int32_t sv1_i =
        static_cast<int32_t>(static_cast<int16_t>(static_cast<uint16_t>(src1.read_lane(wf, lane))));
    if (inst_.abs & (1u << 0))
      sv0 = std::fabs(sv0);
    if (inst_.neg & (1u << 0))
      sv0 = -sv0;
    float result = std::ldexp(sv0, static_cast<int>(sv1_i));
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, util::f32_to_f16(result));
  }
}

VFmaDx9ZeroF32Vop3::VFmaDx9ZeroF32Vop3(const MachineInst *inst)
    : Vop3("v_fma_dx9_zero_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VFmaDx9ZeroF32Vop3::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMadI32I24Vop3::VMadI32I24Vop3(const MachineInst *inst)
    : Vop3("v_mad_i32_i24", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(24, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(24, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMadI32I24Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t a = static_cast<int32_t>(src0.read_lane(wf, lane) << 8) >> 8;
    int32_t b = static_cast<int32_t>(src1.read_lane(wf, lane) << 8) >> 8;
    int32_t c = static_cast<int32_t>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(a * b + c));
  }
}

VMadU32U24Vop3::VMadU32U24Vop3(const MachineInst *inst)
    : Vop3("v_mad_u32_u24", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(24, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(24, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMadU32U24Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane) & 0x00FFFFFFu;
    uint32_t b = src1.read_lane(wf, lane) & 0x00FFFFFFu;
    uint32_t c = src2.read_lane(wf, lane);
    vdst.write_lane(wf, lane, a * b + c);
  }
}

VCubeidF32Vop3::VCubeidF32Vop3(const MachineInst *inst)
    : Vop3("v_cubeid_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VCubeidF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float a = std::bit_cast<float>(src0.read_lane(wf, lane));
    float b = std::bit_cast<float>(src1.read_lane(wf, lane));
    float c = std::bit_cast<float>(src2.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      a = std::fabs(a);
    if (inst_.neg & (1u << 0))
      a = -a;
    if (inst_.abs & (1u << 1))
      b = std::fabs(b);
    if (inst_.neg & (1u << 1))
      b = -b;
    if (inst_.abs & (1u << 2))
      c = std::fabs(c);
    if (inst_.neg & (1u << 2))
      c = -c;
    float ax = std::fabs(a), ay = std::fabs(b), az = std::fabs(c);
    float face;
    if (az >= ax && az >= ay)
      face = c >= 0 ? 4.0f : 5.0f;
    else if (ay >= ax)
      face = b >= 0 ? 2.0f : 3.0f;
    else
      face = a >= 0 ? 0.0f : 1.0f;
    if (inst_.omod == 1)
      face *= 2.0f;
    else if (inst_.omod == 2)
      face *= 4.0f;
    else if (inst_.omod == 3)
      face *= 0.5f;
    if (inst_.clamp)
      face = std::clamp(face, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(face));
  }
}

VCubescF32Vop3::VCubescF32Vop3(const MachineInst *inst)
    : Vop3("v_cubesc_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VCubescF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float a = std::bit_cast<float>(src0.read_lane(wf, lane));
    float b = std::bit_cast<float>(src1.read_lane(wf, lane));
    float c = std::bit_cast<float>(src2.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      a = std::fabs(a);
    if (inst_.neg & (1u << 0))
      a = -a;
    if (inst_.abs & (1u << 1))
      b = std::fabs(b);
    if (inst_.neg & (1u << 1))
      b = -b;
    if (inst_.abs & (1u << 2))
      c = std::fabs(c);
    if (inst_.neg & (1u << 2))
      c = -c;
    float ax = std::fabs(a), ay = std::fabs(b), az = std::fabs(c);
    float sc;
    if (az >= ax && az >= ay)
      sc = c >= 0 ? a : -a;
    else if (ay >= ax)
      sc = a;
    else
      sc = a >= 0 ? -c : c;
    if (inst_.omod == 1)
      sc *= 2.0f;
    else if (inst_.omod == 2)
      sc *= 4.0f;
    else if (inst_.omod == 3)
      sc *= 0.5f;
    if (inst_.clamp)
      sc = std::clamp(sc, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(sc));
  }
}

VCubetcF32Vop3::VCubetcF32Vop3(const MachineInst *inst)
    : Vop3("v_cubetc_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VCubetcF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float a = std::bit_cast<float>(src0.read_lane(wf, lane));
    float b = std::bit_cast<float>(src1.read_lane(wf, lane));
    float c = std::bit_cast<float>(src2.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      a = std::fabs(a);
    if (inst_.neg & (1u << 0))
      a = -a;
    if (inst_.abs & (1u << 1))
      b = std::fabs(b);
    if (inst_.neg & (1u << 1))
      b = -b;
    if (inst_.abs & (1u << 2))
      c = std::fabs(c);
    if (inst_.neg & (1u << 2))
      c = -c;
    float ax = std::fabs(a), ay = std::fabs(b), az = std::fabs(c);
    float tc;
    if (az >= ax && az >= ay)
      tc = -b;
    else if (ay >= ax)
      tc = b >= 0 ? c : -c;
    else
      tc = -b;
    if (inst_.omod == 1)
      tc *= 2.0f;
    else if (inst_.omod == 2)
      tc *= 4.0f;
    else if (inst_.omod == 3)
      tc *= 0.5f;
    if (inst_.clamp)
      tc = std::clamp(tc, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(tc));
  }
}

VCubemaF32Vop3::VCubemaF32Vop3(const MachineInst *inst)
    : Vop3("v_cubema_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VCubemaF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float a = std::bit_cast<float>(src0.read_lane(wf, lane));
    float b = std::bit_cast<float>(src1.read_lane(wf, lane));
    float c = std::bit_cast<float>(src2.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      a = std::fabs(a);
    if (inst_.neg & (1u << 0))
      a = -a;
    if (inst_.abs & (1u << 1))
      b = std::fabs(b);
    if (inst_.neg & (1u << 1))
      b = -b;
    if (inst_.abs & (1u << 2))
      c = std::fabs(c);
    if (inst_.neg & (1u << 2))
      c = -c;
    float ax = std::fabs(a), ay = std::fabs(b), az = std::fabs(c);
    float ma;
    if (az >= ax && az >= ay)
      ma = 2.0f * az;
    else if (ay >= ax)
      ma = 2.0f * ay;
    else
      ma = 2.0f * ax;
    if (inst_.omod == 1)
      ma *= 2.0f;
    else if (inst_.omod == 2)
      ma *= 4.0f;
    else if (inst_.omod == 3)
      ma *= 0.5f;
    if (inst_.clamp)
      ma = std::clamp(ma, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(ma));
  }
}

VBfeU32Vop3::VBfeU32Vop3(const MachineInst *inst)
    : Vop3("v_bfe_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VBfeU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane);
    uint32_t b = src1.read_lane(wf, lane);
    uint32_t c = src2.read_lane(wf, lane);
    uint32_t offset = b & 31;
    uint32_t width = c & 31;
    uint32_t mask = (width == 0 || width >= 32) ? 0u : ((1u << width) - 1);
    vdst.write_lane(wf, lane, width == 0 ? 0u : (a >> offset) & mask);
  }
}

VBfeI32Vop3::VBfeI32Vop3(const MachineInst *inst)
    : Vop3("v_bfe_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VBfeI32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane);
    uint32_t b = src1.read_lane(wf, lane);
    uint32_t c = src2.read_lane(wf, lane);
    uint32_t offset = b & 31;
    uint32_t width = c & 31;
    int32_t sv = static_cast<int32_t>(a);
    int32_t result_val;
    if (width == 0)
      result_val = 0;
    else if (offset + width >= 32)
      result_val = sv >> offset;
    else
      result_val = (sv << (32 - offset - width)) >> (32 - width);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(result_val));
  }
}

VBfiB32Vop3::VBfiB32Vop3(const MachineInst *inst)
    : Vop3("v_bfi_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VBfiB32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane);
    uint32_t b = src1.read_lane(wf, lane);
    uint32_t c = src2.read_lane(wf, lane);
    vdst.write_lane(wf, lane, (a & b) | (~a & c));
  }
}

VFmaF32Vop3::VFmaF32Vop3(const MachineInst *inst)
    : Vop3("v_fma_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VFmaF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float a = std::bit_cast<float>(src0.read_lane(wf, lane));
    float b = std::bit_cast<float>(src1.read_lane(wf, lane));
    float c = std::bit_cast<float>(src2.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      a = std::fabs(a);
    if (inst_.neg & (1u << 0))
      a = -a;
    if (inst_.abs & (1u << 1))
      b = std::fabs(b);
    if (inst_.neg & (1u << 1))
      b = -b;
    if (inst_.abs & (1u << 2))
      c = std::fabs(c);
    if (inst_.neg & (1u << 2))
      c = -c;
    float result = std::fma(a, b, c);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VFmaF64Vop3::VFmaF64Vop3(const MachineInst *inst)
    : Vop3("v_fma_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VFmaF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double a = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double b = std::bit_cast<double>(src1.read_lane64(wf, lane));
    double c = std::bit_cast<double>(src2.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      a = std::fabs(a);
    if (inst_.neg & (1u << 0))
      a = -a;
    if (inst_.abs & (1u << 1))
      b = std::fabs(b);
    if (inst_.neg & (1u << 1))
      b = -b;
    if (inst_.abs & (1u << 2))
      c = std::fabs(c);
    if (inst_.neg & (1u << 2))
      c = -c;
    double result = std::fma(a, b, c);
    if (inst_.omod == 1)
      result *= 2.0;
    else if (inst_.omod == 2)
      result *= 4.0;
    else if (inst_.omod == 3)
      result *= 0.5;
    if (inst_.clamp)
      result = std::clamp(result, 0.0, 1.0);
    vdst.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));
  }
}

VLerpU8Vop3::VLerpU8Vop3(const MachineInst *inst)
    : Vop3("v_lerp_u8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VLerpU8Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane);
    uint32_t b = src1.read_lane(wf, lane);
    uint32_t c = src2.read_lane(wf, lane);
    uint32_t result = 0;
    for (int i = 0; i < 4; ++i) {
      uint32_t ba = (a >> (i * 8)) & 0xFF, bb = (b >> (i * 8)) & 0xFF;
      uint32_t bc = (c >> (i * 8)) & 1;
      result |= (((ba + bb + bc) >> 1) & 0xFF) << (i * 8);
    }
    vdst.write_lane(wf, lane, result);
  }
}

VAlignbitB32Vop3::VAlignbitB32Vop3(const MachineInst *inst)
    : Vop3("v_alignbit_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VAlignbitB32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane);
    uint32_t b = src1.read_lane(wf, lane);
    uint32_t c = src2.read_lane(wf, lane);
    vdst.write_lane(wf, lane,
                    static_cast<uint32_t>(((static_cast<uint64_t>(a) << 32) | b) >> (c & 31)));
  }
}

VAlignbyteB32Vop3::VAlignbyteB32Vop3(const MachineInst *inst)
    : Vop3("v_alignbyte_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VAlignbyteB32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane);
    uint32_t b = src1.read_lane(wf, lane);
    uint32_t c = src2.read_lane(wf, lane);
    vdst.write_lane(wf, lane,
                    static_cast<uint32_t>(((static_cast<uint64_t>(a) << 32) | b) >> ((c & 3) * 8)));
  }
}

VMullitF32Vop3::VMullitF32Vop3(const MachineInst *inst)
    : Vop3("v_mullit_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMullitF32Vop3::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMin3F32Vop3::VMin3F32Vop3(const MachineInst *inst)
    : Vop3("v_min3_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMin3F32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float a = std::bit_cast<float>(src0.read_lane(wf, lane));
    float b = std::bit_cast<float>(src1.read_lane(wf, lane));
    float c = std::bit_cast<float>(src2.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      a = std::fabs(a);
    if (inst_.neg & (1u << 0))
      a = -a;
    if (inst_.abs & (1u << 1))
      b = std::fabs(b);
    if (inst_.neg & (1u << 1))
      b = -b;
    if (inst_.abs & (1u << 2))
      c = std::fabs(c);
    if (inst_.neg & (1u << 2))
      c = -c;
    float result = std::fmin(std::fmin(a, b), c);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VMin3I32Vop3::VMin3I32Vop3(const MachineInst *inst)
    : Vop3("v_min3_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMin3I32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t a = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t b = static_cast<int32_t>(src1.read_lane(wf, lane));
    int32_t c = static_cast<int32_t>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(std::min(std::min(a, b), c)));
  }
}

VMin3U32Vop3::VMin3U32Vop3(const MachineInst *inst)
    : Vop3("v_min3_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMin3U32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane);
    uint32_t b = src1.read_lane(wf, lane);
    uint32_t c = src2.read_lane(wf, lane);
    vdst.write_lane(wf, lane, std::min(std::min(a, b), c));
  }
}

VMax3F32Vop3::VMax3F32Vop3(const MachineInst *inst)
    : Vop3("v_max3_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMax3F32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float a = std::bit_cast<float>(src0.read_lane(wf, lane));
    float b = std::bit_cast<float>(src1.read_lane(wf, lane));
    float c = std::bit_cast<float>(src2.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      a = std::fabs(a);
    if (inst_.neg & (1u << 0))
      a = -a;
    if (inst_.abs & (1u << 1))
      b = std::fabs(b);
    if (inst_.neg & (1u << 1))
      b = -b;
    if (inst_.abs & (1u << 2))
      c = std::fabs(c);
    if (inst_.neg & (1u << 2))
      c = -c;
    float result = std::fmax(std::fmax(a, b), c);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VMax3I32Vop3::VMax3I32Vop3(const MachineInst *inst)
    : Vop3("v_max3_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMax3I32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t a = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t b = static_cast<int32_t>(src1.read_lane(wf, lane));
    int32_t c = static_cast<int32_t>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(std::max(std::max(a, b), c)));
  }
}

VMax3U32Vop3::VMax3U32Vop3(const MachineInst *inst)
    : Vop3("v_max3_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMax3U32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane);
    uint32_t b = src1.read_lane(wf, lane);
    uint32_t c = src2.read_lane(wf, lane);
    vdst.write_lane(wf, lane, std::max(std::max(a, b), c));
  }
}

VMed3F32Vop3::VMed3F32Vop3(const MachineInst *inst)
    : Vop3("v_med3_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMed3F32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float a = std::bit_cast<float>(src0.read_lane(wf, lane));
    float b = std::bit_cast<float>(src1.read_lane(wf, lane));
    float c = std::bit_cast<float>(src2.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      a = std::fabs(a);
    if (inst_.neg & (1u << 0))
      a = -a;
    if (inst_.abs & (1u << 1))
      b = std::fabs(b);
    if (inst_.neg & (1u << 1))
      b = -b;
    if (inst_.abs & (1u << 2))
      c = std::fabs(c);
    if (inst_.neg & (1u << 2))
      c = -c;
    float result = std::fmax(std::fmin(std::fmax(a, b), c), std::fmin(a, b));
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VMed3I32Vop3::VMed3I32Vop3(const MachineInst *inst)
    : Vop3("v_med3_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMed3I32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t a = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t b = static_cast<int32_t>(src1.read_lane(wf, lane));
    int32_t c = static_cast<int32_t>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane,
                    static_cast<uint32_t>(std::max(std::min(std::max(a, b), c), std::min(a, b))));
  }
}

VMed3U32Vop3::VMed3U32Vop3(const MachineInst *inst)
    : Vop3("v_med3_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMed3U32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane);
    uint32_t b = src1.read_lane(wf, lane);
    uint32_t c = src2.read_lane(wf, lane);
    vdst.write_lane(wf, lane, std::max(std::min(std::max(a, b), c), std::min(a, b)));
  }
}

VSadU8Vop3::VSadU8Vop3(const MachineInst *inst)
    : Vop3("v_sad_u8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSadU8Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane);
    uint32_t b = src1.read_lane(wf, lane);
    uint32_t c = src2.read_lane(wf, lane);
    uint32_t sum = 0;
    for (int i = 0; i < 4; ++i) {
      uint32_t ba = (a >> (i * 8)) & 0xFF, bb = (b >> (i * 8)) & 0xFF;
      sum += ba > bb ? ba - bb : bb - ba;
    }
    vdst.write_lane(wf, lane, sum + c);
  }
}

VSadHiU8Vop3::VSadHiU8Vop3(const MachineInst *inst)
    : Vop3("v_sad_hi_u8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSadHiU8Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane);
    uint32_t b = src1.read_lane(wf, lane);
    uint32_t c = src2.read_lane(wf, lane);
    uint32_t sum = 0;
    for (int i = 0; i < 4; ++i) {
      uint32_t ba = (a >> (i * 8)) & 0xFF, bb = (b >> (i * 8)) & 0xFF;
      sum += ba > bb ? ba - bb : bb - ba;
    }
    vdst.write_lane(wf, lane, (sum << 16) + c);
  }
}

VSadU16Vop3::VSadU16Vop3(const MachineInst *inst)
    : Vop3("v_sad_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSadU16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane);
    uint32_t b = src1.read_lane(wf, lane);
    uint32_t c = src2.read_lane(wf, lane);
    uint32_t lo_a = a & 0xFFFF, hi_a = a >> 16;
    uint32_t lo_b = b & 0xFFFF, hi_b = b >> 16;
    uint32_t sum =
        (lo_a > lo_b ? lo_a - lo_b : lo_b - lo_a) + (hi_a > hi_b ? hi_a - hi_b : hi_b - hi_a);
    vdst.write_lane(wf, lane, sum + c);
  }
}

VSadU32Vop3::VSadU32Vop3(const MachineInst *inst)
    : Vop3("v_sad_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VSadU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane);
    uint32_t b = src1.read_lane(wf, lane);
    uint32_t c = src2.read_lane(wf, lane);
    vdst.write_lane(wf, lane, (a > b ? a - b : b - a) + c);
  }
}

VCvtPkU8F32Vop3::VCvtPkU8F32Vop3(const MachineInst *inst)
    : Vop3("v_cvt_pk_u8_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VCvtPkU8F32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float fval = std::bit_cast<float>(src0.read_lane(wf, lane));
    uint32_t byte_sel = src1.read_lane(wf, lane) & 3;
    uint32_t old = src2.read_lane(wf, lane);
    uint32_t byte = static_cast<uint32_t>(std::clamp(fval, 0.0f, 255.0f));
    uint32_t mask = ~(0xFFu << (byte_sel * 8));
    vdst.write_lane(wf, lane, (old & mask) | (byte << (byte_sel * 8)));
  }
}

VDivFixupF32Vop3::VDivFixupF32Vop3(const MachineInst *inst)
    : Vop3("v_div_fixup_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VDivFixupF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float p = std::bit_cast<float>(src0.read_lane(wf, lane));
    float b = std::bit_cast<float>(src1.read_lane(wf, lane));
    float c = std::bit_cast<float>(src2.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      p = std::fabs(p);
    if (inst_.neg & (1u << 0))
      p = -p;
    if (inst_.abs & (1u << 1))
      b = std::fabs(b);
    if (inst_.neg & (1u << 1))
      b = -b;
    if (inst_.abs & (1u << 2))
      c = std::fabs(c);
    if (inst_.neg & (1u << 2))
      c = -c;
    float result;
    if (std::isnan(b))
      result = b;
    else if (std::isnan(c))
      result = c;
    else if (c == 0.0f && b == 0.0f)
      result = std::numeric_limits<float>::quiet_NaN();
    else if (std::isinf(c) && std::isinf(b))
      result = std::numeric_limits<float>::quiet_NaN();
    else if (c == 0.0f) {
      result = std::copysign(
          std::numeric_limits<float>::infinity(),
          std::bit_cast<float>(std::bit_cast<uint32_t>(b) ^ std::bit_cast<uint32_t>(c)));
    } else if (std::isinf(b))
      result = std::copysign(0.0f, b);
    else
      result = p;
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VDivFixupF64Vop3::VDivFixupF64Vop3(const MachineInst *inst)
    : Vop3("v_div_fixup_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VDivFixupF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double p = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double b = std::bit_cast<double>(src1.read_lane64(wf, lane));
    double c = std::bit_cast<double>(src2.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      p = std::fabs(p);
    if (inst_.neg & (1u << 0))
      p = -p;
    if (inst_.abs & (1u << 1))
      b = std::fabs(b);
    if (inst_.neg & (1u << 1))
      b = -b;
    if (inst_.abs & (1u << 2))
      c = std::fabs(c);
    if (inst_.neg & (1u << 2))
      c = -c;
    double result;
    if (std::isnan(b))
      result = b;
    else if (std::isnan(c))
      result = c;
    else if (c == 0.0 && b == 0.0)
      result = std::numeric_limits<double>::quiet_NaN();
    else if (std::isinf(c) && std::isinf(b))
      result = std::numeric_limits<double>::quiet_NaN();
    else if (c == 0.0) {
      result = std::copysign(
          std::numeric_limits<double>::infinity(),
          std::bit_cast<double>(std::bit_cast<uint64_t>(b) ^ std::bit_cast<uint64_t>(c)));
    } else if (std::isinf(b))
      result = std::copysign(0.0, b);
    else
      result = p;
    if (inst_.omod == 1)
      result *= 2.0;
    else if (inst_.omod == 2)
      result *= 4.0;
    else if (inst_.omod == 3)
      result *= 0.5;
    if (inst_.clamp)
      result = std::clamp(result, 0.0, 1.0);
    vdst.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));
  }
}

VDivFmasF32Vop3::VDivFmasF32Vop3(const MachineInst *inst)
    : Vop3("v_div_fmas_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VDivFmasF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    float s2 = std::bit_cast<float>(src2.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (inst_.abs & (1u << 2))
      s2 = std::fabs(s2);
    if (inst_.neg & (1u << 2))
      s2 = -s2;
    float result = std::fma(s0, s1, s2);
    if (vcc & (1ULL << lane)) {
      result = std::ldexp(result, 128);
    }
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VDivFmasF64Vop3::VDivFmasF64Vop3(const MachineInst *inst)
    : Vop3("v_div_fmas_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VDivFmasF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    double s2 = std::bit_cast<double>(src2.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (inst_.abs & (1u << 2))
      s2 = std::fabs(s2);
    if (inst_.neg & (1u << 2))
      s2 = -s2;
    double result = std::fma(s0, s1, s2);
    if (vcc & (1ULL << lane)) {
      result = std::ldexp(result, 1024);
    }
    vdst.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));
  }
}

VMsadU8Vop3::VMsadU8Vop3(const MachineInst *inst)
    : Vop3("v_msad_u8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMsadU8Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane);
    uint32_t b = src1.read_lane(wf, lane);
    uint32_t c = src2.read_lane(wf, lane);
    uint32_t sum = 0;
    for (int i = 0; i < 4; ++i) {
      uint32_t ba = (a >> (i * 8)) & 0xFF, bb = (b >> (i * 8)) & 0xFF;
      if (ba != 0)
        sum += ba > bb ? ba - bb : bb - ba;
    }
    vdst.write_lane(wf, lane, sum + c);
  }
}

VQsadPkU16U8Vop3::VQsadPkU16U8Vop3(const MachineInst *inst)
    : Vop3("v_qsad_pk_u16_u8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VQsadPkU16U8Vop3::execute(amdgpu::Wavefront &wf) { (void)wf; }

VMqsadPkU16U8Vop3::VMqsadPkU16U8Vop3(const MachineInst *inst)
    : Vop3("v_mqsad_pk_u16_u8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMqsadPkU16U8Vop3::execute(amdgpu::Wavefront &wf) { (void)wf; }

VMqsadU32U8Vop3::VMqsadU32U8Vop3(const MachineInst *inst)
    : Vop3("v_mqsad_u32_u8", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(128, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(128, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMqsadU32U8Vop3::execute(amdgpu::Wavefront &wf) { (void)wf; }

VXor3B32Vop3::VXor3B32Vop3(const MachineInst *inst)
    : Vop3("v_xor3_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VXor3B32Vop3::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMadU16Vop3::VMadU16Vop3(const MachineInst *inst)
    : Vop3("v_mad_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMadU16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t a = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t b = static_cast<uint16_t>(src1.read_lane(wf, lane));
    uint16_t c = static_cast<uint16_t>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(static_cast<uint16_t>(a * b + c)));
  }
}

VPermB32Vop3::VPermB32Vop3(const MachineInst *inst)
    : Vop3("v_perm_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VPermB32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane);
    uint32_t b = src1.read_lane(wf, lane);
    uint32_t c = src2.read_lane(wf, lane);
    uint32_t result = 0;
    uint64_t src = (static_cast<uint64_t>(a) << 32) | b;
    for (int i = 0; i < 4; ++i) {
      uint32_t sel = (c >> (i * 8)) & 0xFF;
      uint32_t byte;
      if (sel <= 7)
        byte = (src >> (sel * 8)) & 0xFF;
      else if (sel >= 0x09 && sel <= 0x0C) {
        uint32_t bi = sel - 0x09;
        byte = ((src >> (bi * 8 + 7)) & 1) ? 0xFF : 0x00;
      } else if (sel == 0x0D)
        byte = 0xFF;
      else
        byte = 0;
      result |= byte << (i * 8);
    }
    vdst.write_lane(wf, lane, result);
  }
}

VXadU32Vop3::VXadU32Vop3(const MachineInst *inst)
    : Vop3("v_xad_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VXadU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane);
    uint32_t b = src1.read_lane(wf, lane);
    uint32_t c = src2.read_lane(wf, lane);
    vdst.write_lane(wf, lane, (a ^ b) + c);
  }
}

VLshlAddU32Vop3::VLshlAddU32Vop3(const MachineInst *inst)
    : Vop3("v_lshl_add_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VLshlAddU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane);
    uint32_t b = src1.read_lane(wf, lane);
    uint32_t c = src2.read_lane(wf, lane);
    vdst.write_lane(wf, lane, (a << (b & 31)) + c);
  }
}

VAddLshlU32Vop3::VAddLshlU32Vop3(const MachineInst *inst)
    : Vop3("v_add_lshl_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VAddLshlU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane);
    uint32_t b = src1.read_lane(wf, lane);
    uint32_t c = src2.read_lane(wf, lane);
    vdst.write_lane(wf, lane, (a + b) << (c & 31));
  }
}

VFmaF16Vop3::VFmaF16Vop3(const MachineInst *inst)
    : Vop3("v_fma_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VFmaF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float a = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float b = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    float c = util::f16_to_f32(static_cast<uint16_t>(src2.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      a = std::fabs(a);
    if (inst_.neg & (1u << 0))
      a = -a;
    if (inst_.abs & (1u << 1))
      b = std::fabs(b);
    if (inst_.neg & (1u << 1))
      b = -b;
    if (inst_.abs & (1u << 2))
      c = std::fabs(c);
    if (inst_.neg & (1u << 2))
      c = -c;
    float result = std::fma(a, b, c);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, util::f32_to_f16(result));
  }
}

VMin3F16Vop3::VMin3F16Vop3(const MachineInst *inst)
    : Vop3("v_min3_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMin3F16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float a = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float b = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    float c = util::f16_to_f32(static_cast<uint16_t>(src2.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      a = std::fabs(a);
    if (inst_.neg & (1u << 0))
      a = -a;
    if (inst_.abs & (1u << 1))
      b = std::fabs(b);
    if (inst_.neg & (1u << 1))
      b = -b;
    if (inst_.abs & (1u << 2))
      c = std::fabs(c);
    if (inst_.neg & (1u << 2))
      c = -c;
    float result = std::fmin(std::fmin(a, b), c);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, util::f32_to_f16(result));
  }
}

VMin3I16Vop3::VMin3I16Vop3(const MachineInst *inst)
    : Vop3("v_min3_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMin3I16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t a = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t b = static_cast<int16_t>(src1.read_lane(wf, lane) & 0xFFFF);
    int16_t c = static_cast<int16_t>(src2.read_lane(wf, lane) & 0xFFFF);
    vdst.write_lane(wf, lane,
                    static_cast<uint32_t>(static_cast<uint16_t>(std::min(std::min(a, b), c))));
  }
}

VMin3U16Vop3::VMin3U16Vop3(const MachineInst *inst)
    : Vop3("v_min3_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMin3U16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t a = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t b = static_cast<uint16_t>(src1.read_lane(wf, lane));
    uint16_t c = static_cast<uint16_t>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(std::min(std::min(a, b), c)));
  }
}

VMax3F16Vop3::VMax3F16Vop3(const MachineInst *inst)
    : Vop3("v_max3_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMax3F16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float a = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float b = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    float c = util::f16_to_f32(static_cast<uint16_t>(src2.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      a = std::fabs(a);
    if (inst_.neg & (1u << 0))
      a = -a;
    if (inst_.abs & (1u << 1))
      b = std::fabs(b);
    if (inst_.neg & (1u << 1))
      b = -b;
    if (inst_.abs & (1u << 2))
      c = std::fabs(c);
    if (inst_.neg & (1u << 2))
      c = -c;
    float result = std::fmax(std::fmax(a, b), c);
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, util::f32_to_f16(result));
  }
}

VMax3I16Vop3::VMax3I16Vop3(const MachineInst *inst)
    : Vop3("v_max3_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMax3I16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t a = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t b = static_cast<int16_t>(src1.read_lane(wf, lane) & 0xFFFF);
    int16_t c = static_cast<int16_t>(src2.read_lane(wf, lane) & 0xFFFF);
    vdst.write_lane(wf, lane,
                    static_cast<uint32_t>(static_cast<uint16_t>(std::max(std::max(a, b), c))));
  }
}

VMax3U16Vop3::VMax3U16Vop3(const MachineInst *inst)
    : Vop3("v_max3_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMax3U16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t a = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t b = static_cast<uint16_t>(src1.read_lane(wf, lane));
    uint16_t c = static_cast<uint16_t>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(std::max(std::max(a, b), c)));
  }
}

VMed3F16Vop3::VMed3F16Vop3(const MachineInst *inst)
    : Vop3("v_med3_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMed3F16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float a = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float b = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    float c = util::f16_to_f32(static_cast<uint16_t>(src2.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      a = std::fabs(a);
    if (inst_.neg & (1u << 0))
      a = -a;
    if (inst_.abs & (1u << 1))
      b = std::fabs(b);
    if (inst_.neg & (1u << 1))
      b = -b;
    if (inst_.abs & (1u << 2))
      c = std::fabs(c);
    if (inst_.neg & (1u << 2))
      c = -c;
    float result = std::fmax(std::fmin(std::fmax(a, b), c), std::fmin(a, b));
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, util::f32_to_f16(result));
  }
}

VMed3I16Vop3::VMed3I16Vop3(const MachineInst *inst)
    : Vop3("v_med3_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMed3I16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t a = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t b = static_cast<int16_t>(src1.read_lane(wf, lane) & 0xFFFF);
    int16_t c = static_cast<int16_t>(src2.read_lane(wf, lane) & 0xFFFF);
    vdst.write_lane(wf, lane,
                    static_cast<uint32_t>(static_cast<uint16_t>(
                        std::max(std::min(std::max(a, b), c), std::min(a, b)))));
  }
}

VMed3U16Vop3::VMed3U16Vop3(const MachineInst *inst)
    : Vop3("v_med3_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMed3U16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t a = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t b = static_cast<uint16_t>(src1.read_lane(wf, lane));
    uint16_t c = static_cast<uint16_t>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane,
                    static_cast<uint32_t>(std::max(std::min(std::max(a, b), c), std::min(a, b))));
  }
}

VMadI16Vop3::VMadI16Vop3(const MachineInst *inst)
    : Vop3("v_mad_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMadI16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t a = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t b = static_cast<int16_t>(src1.read_lane(wf, lane) & 0xFFFF);
    int16_t c = static_cast<int16_t>(src2.read_lane(wf, lane) & 0xFFFF);
    vdst.write_lane(wf, lane,
                    static_cast<uint32_t>(static_cast<uint16_t>(static_cast<int16_t>(a * b + c))));
  }
}

VDivFixupF16Vop3::VDivFixupF16Vop3(const MachineInst *inst)
    : Vop3("v_div_fixup_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VDivFixupF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float p = std::bit_cast<float>(src0.read_lane(wf, lane));
    float b = std::bit_cast<float>(src1.read_lane(wf, lane));
    float c = std::bit_cast<float>(src2.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      p = std::fabs(p);
    if (inst_.neg & (1u << 0))
      p = -p;
    if (inst_.abs & (1u << 1))
      b = std::fabs(b);
    if (inst_.neg & (1u << 1))
      b = -b;
    if (inst_.abs & (1u << 2))
      c = std::fabs(c);
    if (inst_.neg & (1u << 2))
      c = -c;
    float result;
    if (std::isnan(b))
      result = b;
    else if (std::isnan(c))
      result = c;
    else if (c == 0.0f && b == 0.0f)
      result = std::numeric_limits<float>::quiet_NaN();
    else if (std::isinf(c) && std::isinf(b))
      result = std::numeric_limits<float>::quiet_NaN();
    else if (c == 0.0f) {
      result = std::copysign(
          std::numeric_limits<float>::infinity(),
          std::bit_cast<float>(std::bit_cast<uint32_t>(b) ^ std::bit_cast<uint32_t>(c)));
    } else if (std::isinf(b))
      result = std::copysign(0.0f, b);
    else
      result = p;
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VAdd3U32Vop3::VAdd3U32Vop3(const MachineInst *inst)
    : Vop3("v_add3_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VAdd3U32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane);
    uint32_t b = src1.read_lane(wf, lane);
    uint32_t c = src2.read_lane(wf, lane);
    vdst.write_lane(wf, lane, a + b + c);
  }
}

VLshlOrB32Vop3::VLshlOrB32Vop3(const MachineInst *inst)
    : Vop3("v_lshl_or_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VLshlOrB32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane);
    uint32_t b = src1.read_lane(wf, lane);
    uint32_t c = src2.read_lane(wf, lane);
    vdst.write_lane(wf, lane, (a << (b & 31)) | c);
  }
}

VAndOrB32Vop3::VAndOrB32Vop3(const MachineInst *inst)
    : Vop3("v_and_or_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VAndOrB32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane);
    uint32_t b = src1.read_lane(wf, lane);
    uint32_t c = src2.read_lane(wf, lane);
    vdst.write_lane(wf, lane, (a & b) | c);
  }
}

VOr3B32Vop3::VOr3B32Vop3(const MachineInst *inst)
    : Vop3("v_or3_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VOr3B32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t a = src0.read_lane(wf, lane);
    uint32_t b = src1.read_lane(wf, lane);
    uint32_t c = src2.read_lane(wf, lane);
    vdst.write_lane(wf, lane, a | b | c);
  }
}

VMadU32U16Vop3::VMadU32U16Vop3(const MachineInst *inst)
    : Vop3("v_mad_u32_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMadU32U16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane) & 0xFFFFu;
    uint32_t s1 = src1.read_lane(wf, lane) & 0xFFFFu;
    uint32_t s2 = src2.read_lane(wf, lane);
    vdst.write_lane(wf, lane, s0 * s1 + s2);
  }
}

VMadI32I16Vop3::VMadI32I16Vop3(const MachineInst *inst)
    : Vop3("v_mad_i32_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMadI32I16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int32_t s1 = static_cast<int16_t>(src1.read_lane(wf, lane) & 0xFFFF);
    int32_t s2 = static_cast<int32_t>(src2.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(s0 * s1 + s2));
  }
}

VPermlane16B32Vop3::VPermlane16B32Vop3(const MachineInst *inst)
    : Vop3("v_permlane16_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VPermlane16B32Vop3::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VPermlanex16B32Vop3::VPermlanex16B32Vop3(const MachineInst *inst)
    : Vop3("v_permlanex16_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VPermlanex16B32Vop3::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VCndmaskB16Vop3::VCndmaskB16Vop3(const MachineInst *inst)
    : Vop3("v_cndmask_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VCndmaskB16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    float val = (vcc & (1ULL << lane)) ? s1 : s0;
    if (inst_.omod == 1)
      val *= 2.0f;
    else if (inst_.omod == 2)
      val *= 4.0f;
    else if (inst_.omod == 3)
      val *= 0.5f;
    if (inst_.clamp)
      val = std::clamp(val, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(val));
  }
}

VMaxminF32Vop3::VMaxminF32Vop3(const MachineInst *inst)
    : Vop3("v_maxmin_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMaxminF32Vop3::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMinmaxF32Vop3::VMinmaxF32Vop3(const MachineInst *inst)
    : Vop3("v_minmax_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMinmaxF32Vop3::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMaxminF16Vop3::VMaxminF16Vop3(const MachineInst *inst)
    : Vop3("v_maxmin_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMaxminF16Vop3::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMinmaxF16Vop3::VMinmaxF16Vop3(const MachineInst *inst)
    : Vop3("v_minmax_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMinmaxF16Vop3::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMaxminU32Vop3::VMaxminU32Vop3(const MachineInst *inst)
    : Vop3("v_maxmin_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMaxminU32Vop3::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMinmaxU32Vop3::VMinmaxU32Vop3(const MachineInst *inst)
    : Vop3("v_minmax_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMinmaxU32Vop3::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMaxminI32Vop3::VMaxminI32Vop3(const MachineInst *inst)
    : Vop3("v_maxmin_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMaxminI32Vop3::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMinmaxI32Vop3::VMinmaxI32Vop3(const MachineInst *inst)
    : Vop3("v_minmax_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMinmaxI32Vop3::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VDot2F16F16Vop3::VDot2F16F16Vop3(const MachineInst *inst)
    : Vop3("v_dot2_f16_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VDot2F16F16Vop3::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VDot2Bf16Bf16Vop3::VDot2Bf16Bf16Vop3(const MachineInst *inst)
    : Vop3("v_dot2_bf16_bf16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VDot2Bf16Bf16Vop3::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VDivScaleF32Vop3SdstEnc::VDivScaleF32Vop3SdstEnc(const MachineInst *inst)
    : Vop3SdstEnc("v_div_scale_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      sdst(64, OperandType::OPR_SDST, 0),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VDivScaleF32Vop3SdstEnc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    float s2 = std::bit_cast<float>(src2.read_lane(wf, lane));
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (inst_.neg & (1u << 2))
      s2 = -s2;
    float result = s0;
    bool needs_scale = false;
    if (!std::isnan(s1) && !std::isnan(s2) && !std::isinf(s1) && !std::isinf(s2) && s1 != 0.0f &&
        s2 != 0.0f) {
      int exp1, exp2;
      std::frexp(s1, &exp1);
      std::frexp(s2, &exp2);
      needs_scale = std::abs(exp1 - exp2) > 100;
      if (needs_scale)
        result = std::ldexp(s0, exp2 > exp1 ? 128 : -128);
    }
    if (needs_scale)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
  wf.set_vcc(vcc);
}

VDivScaleF64Vop3SdstEnc::VDivScaleF64Vop3SdstEnc(const MachineInst *inst)
    : Vop3SdstEnc("v_div_scale_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      sdst(64, OperandType::OPR_SDST, 0),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VDivScaleF64Vop3SdstEnc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    double s2 = std::bit_cast<double>(src2.read_lane64(wf, lane));
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (inst_.neg & (1u << 2))
      s2 = -s2;
    double result = s0;
    bool needs_scale = false;
    if (!std::isnan(s1) && !std::isnan(s2) && !std::isinf(s1) && !std::isinf(s2) && s1 != 0.0 &&
        s2 != 0.0) {
      int exp1, exp2;
      std::frexp(s1, &exp1);
      std::frexp(s2, &exp2);
      needs_scale = std::abs(exp1 - exp2) > 768;
      if (needs_scale)
        result = std::ldexp(s0, exp2 > exp1 ? 1024 : -1024);
    }
    if (needs_scale)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
    vdst.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));
  }
  wf.set_vcc(vcc);
}

VMadU64U32Vop3SdstEnc::VMadU64U32Vop3SdstEnc(const MachineInst *inst)
    : Vop3SdstEnc("v_mad_u64_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      sdst(64, OperandType::OPR_SREG, 0),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMadU64U32Vop3SdstEnc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t s0 = src0.read_lane(wf, lane);
    uint64_t s1 = src1.read_lane(wf, lane);
    uint64_t s2 = src2.read_lane64(wf, lane);
    uint64_t result = s0 * s1 + s2;
    vdst.write_lane64(wf, lane, result);
  }
}

VMadI64I32Vop3SdstEnc::VMadI64I32Vop3SdstEnc(const MachineInst *inst)
    : Vop3SdstEnc("v_mad_i64_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      sdst(64, OperandType::OPR_SREG, 0),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1),
      src2(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src2) {
  dst_operands_.emplace_back(&vdst);
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
  src_operands_.emplace_back(&src2);
}

void VMadI64I32Vop3SdstEnc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int64_t s0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int64_t s1 = static_cast<int32_t>(src1.read_lane(wf, lane));
    int64_t s2 = static_cast<int64_t>(src2.read_lane64(wf, lane));
    uint64_t result = static_cast<uint64_t>(s0 * s1 + s2);
    vdst.write_lane64(wf, lane, result);
  }
}

VAddCoU32Vop3SdstEnc::VAddCoU32Vop3SdstEnc(const MachineInst *inst)
    : Vop3SdstEnc("v_add_co_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      sdst(64, OperandType::OPR_SREG, 0),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VAddCoU32Vop3SdstEnc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = src1.read_lane(wf, lane);
    uint64_t wide = static_cast<uint64_t>(sv0) + static_cast<uint64_t>(sv1);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(wide));
    if (wide > 0xFFFFFFFFULL)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VSubCoU32Vop3SdstEnc::VSubCoU32Vop3SdstEnc(const MachineInst *inst)
    : Vop3SdstEnc("v_sub_co_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      sdst(64, OperandType::OPR_SREG, 0),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VSubCoU32Vop3SdstEnc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = src1.read_lane(wf, lane);
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

VSubrevCoU32Vop3SdstEnc::VSubrevCoU32Vop3SdstEnc(const MachineInst *inst)
    : Vop3SdstEnc("v_subrev_co_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      sdst(64, OperandType::OPR_SREG, 0),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  dst_operands_.emplace_back(&sdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VSubrevCoU32Vop3SdstEnc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = src1.read_lane(wf, lane);
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

VAddNcU16Vop3::VAddNcU16Vop3(const MachineInst *inst)
    : Vop3("v_add_nc_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VAddNcU16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t sv0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t sv1 = static_cast<uint16_t>(src1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(static_cast<uint16_t>(sv0 + sv1)));
  }
}

VSubNcU16Vop3::VSubNcU16Vop3(const MachineInst *inst)
    : Vop3("v_sub_nc_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VSubNcU16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t sv0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t sv1 = static_cast<uint16_t>(src1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(static_cast<uint16_t>(sv0 - sv1)));
  }
}

VMulLoU16Vop3::VMulLoU16Vop3(const MachineInst *inst)
    : Vop3("v_mul_lo_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMulLoU16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t sv0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t sv1 = static_cast<uint16_t>(src1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(static_cast<uint16_t>(sv0 * sv1)));
  }
}

VCvtPkI16F32Vop3::VCvtPkI16F32Vop3(const MachineInst *inst)
    : Vop3("v_cvt_pk_i16_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCvtPkI16F32Vop3::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VCvtPkU16F32Vop3::VCvtPkU16F32Vop3(const MachineInst *inst)
    : Vop3("v_cvt_pk_u16_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCvtPkU16F32Vop3::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VMaxU16Vop3::VMaxU16Vop3(const MachineInst *inst)
    : Vop3("v_max_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMaxU16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t sv0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t sv1 = static_cast<uint16_t>(src1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(sv0 > sv1 ? sv0 : sv1));
  }
}

VMaxI16Vop3::VMaxI16Vop3(const MachineInst *inst)
    : Vop3("v_max_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMaxI16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t sv0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t sv1 = static_cast<int16_t>(src1.read_lane(wf, lane) & 0xFFFF);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(static_cast<uint16_t>(sv0 > sv1 ? sv0 : sv1)));
  }
}

VMinU16Vop3::VMinU16Vop3(const MachineInst *inst)
    : Vop3("v_min_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMinU16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t sv0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t sv1 = static_cast<uint16_t>(src1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(sv0 < sv1 ? sv0 : sv1));
  }
}

VMinI16Vop3::VMinI16Vop3(const MachineInst *inst)
    : Vop3("v_min_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMinI16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t sv0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t sv1 = static_cast<int16_t>(src1.read_lane(wf, lane) & 0xFFFF);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(static_cast<uint16_t>(sv0 < sv1 ? sv0 : sv1)));
  }
}

VAddNcI16Vop3::VAddNcI16Vop3(const MachineInst *inst)
    : Vop3("v_add_nc_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VAddNcI16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t sv0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t sv1 = static_cast<int16_t>(src1.read_lane(wf, lane) & 0xFFFF);
    vdst.write_lane(wf, lane,
                    static_cast<uint32_t>(static_cast<uint16_t>(static_cast<int16_t>(sv0 + sv1))));
  }
}

VSubNcI16Vop3::VSubNcI16Vop3(const MachineInst *inst)
    : Vop3("v_sub_nc_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VSubNcI16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t sv0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t sv1 = static_cast<int16_t>(src1.read_lane(wf, lane) & 0xFFFF);
    vdst.write_lane(wf, lane,
                    static_cast<uint32_t>(static_cast<uint16_t>(static_cast<int16_t>(sv0 - sv1))));
  }
}

VPackB32F16Vop3::VPackB32F16Vop3(const MachineInst *inst)
    : Vop3("v_pack_b32_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VPackB32F16Vop3::execute(amdgpu::Wavefront &wf) { (void)wf; }

VCvtPkNormI16F16Vop3::VCvtPkNormI16F16Vop3(const MachineInst *inst)
    : Vop3("v_cvt_pk_norm_i16_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCvtPkNormI16F16Vop3::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VCvtPkNormU16F16Vop3::VCvtPkNormU16F16Vop3(const MachineInst *inst)
    : Vop3("v_cvt_pk_norm_u16_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCvtPkNormU16F16Vop3::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VLdexpF32Vop3::VLdexpF32Vop3(const MachineInst *inst)
    : Vop3("v_ldexp_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VLdexpF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float sv0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    int32_t sv1_i = static_cast<int32_t>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      sv0 = std::fabs(sv0);
    if (inst_.neg & (1u << 0))
      sv0 = -sv0;
    float result = std::ldexp(sv0, static_cast<int>(sv1_i));
    if (inst_.omod == 1)
      result *= 2.0f;
    else if (inst_.omod == 2)
      result *= 4.0f;
    else if (inst_.omod == 3)
      result *= 0.5f;
    if (inst_.clamp)
      result = std::clamp(result, 0.0f, 1.0f);
    vdst.write_lane(wf, lane, std::bit_cast<uint32_t>(result));
  }
}

VBfmB32Vop3::VBfmB32Vop3(const MachineInst *inst)
    : Vop3("v_bfm_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VBfmB32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = src1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, (sv0 & 31u) == 0 ? 0u : ((1u << (sv0 & 31u)) - 1u) << (sv1 & 31u));
  }
}

VBcntU32B32Vop3::VBcntU32B32Vop3(const MachineInst *inst)
    : Vop3("v_bcnt_u32_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VBcntU32B32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s = src0.read_lane(wf, lane);
    vdst.write_lane(wf, lane, static_cast<uint32_t>(std::popcount(s)));
  }
}

VMbcntLoU32B32Vop3::VMbcntLoU32B32Vop3(const MachineInst *inst)
    : Vop3("v_mbcnt_lo_u32_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMbcntLoU32B32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t mask = src0.read_lane(wf, lane);
    uint32_t base = src1.read_lane(wf, lane);
    uint32_t thread_mask = lane < 32 ? (1u << lane) - 1 : 0xFFFFFFFFu;
    uint32_t count = std::popcount(mask & thread_mask);
    vdst.write_lane(wf, lane, base + count);
  }
}

VMbcntHiU32B32Vop3::VMbcntHiU32B32Vop3(const MachineInst *inst)
    : Vop3("v_mbcnt_hi_u32_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMbcntHiU32B32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t mask = src0.read_lane(wf, lane);
    uint32_t base = src1.read_lane(wf, lane);
    uint32_t shift = lane >= 32 ? lane - 32 : 0;
    uint32_t thread_mask = lane >= 32 ? (1u << shift) - 1 : 0;
    uint32_t count = std::popcount(mask & thread_mask);
    vdst.write_lane(wf, lane, base + count);
  }
}

VCvtPkNormI16F32Vop3::VCvtPkNormI16F32Vop3(const MachineInst *inst)
    : Vop3("v_cvt_pk_norm_i16_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCvtPkNormI16F32Vop3::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VCvtPkNormU16F32Vop3::VCvtPkNormU16F32Vop3(const MachineInst *inst)
    : Vop3("v_cvt_pk_norm_u16_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCvtPkNormU16F32Vop3::execute(amdgpu::Wavefront &wf) {
  (void)wf;
  throw util::UnimplementedInst(mnemonic());
}

VCvtPkU16U32Vop3::VCvtPkU16U32Vop3(const MachineInst *inst)
    : Vop3("v_cvt_pk_u16_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCvtPkU16U32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane);
    uint32_t s1 = src1.read_lane(wf, lane);
    uint16_t lo = static_cast<uint16_t>(std::min(s0, 0xFFFFu));
    uint16_t hi = static_cast<uint16_t>(std::min(s1, 0xFFFFu));
    vdst.write_lane(wf, lane,
                    (static_cast<uint32_t>(static_cast<uint16_t>(hi)) << 16) |
                        static_cast<uint32_t>(static_cast<uint16_t>(lo)));
  }
}

VCvtPkI16I32Vop3::VCvtPkI16I32Vop3(const MachineInst *inst)
    : Vop3("v_cvt_pk_i16_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCvtPkI16I32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane);
    uint32_t s1 = src1.read_lane(wf, lane);
    int16_t lo = static_cast<int16_t>(std::clamp(static_cast<int32_t>(s0), -32768, 32767));
    int16_t hi = static_cast<int16_t>(std::clamp(static_cast<int32_t>(s1), -32768, 32767));
    vdst.write_lane(wf, lane,
                    (static_cast<uint32_t>(static_cast<uint16_t>(hi)) << 16) |
                        static_cast<uint32_t>(static_cast<uint16_t>(lo)));
  }
}

VSubNcI32Vop3::VSubNcI32Vop3(const MachineInst *inst)
    : Vop3("v_sub_nc_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VSubNcI32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t sv0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t sv1 = static_cast<int32_t>(src1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(sv0 - sv1));
  }
}

VAddNcI32Vop3::VAddNcI32Vop3(const MachineInst *inst)
    : Vop3("v_add_nc_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VAddNcI32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t sv0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t sv1 = static_cast<int32_t>(src1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(sv0 + sv1));
  }
}

VAddF64Vop3::VAddF64Vop3(const MachineInst *inst)
    : Vop3("v_add_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VAddF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double sv0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double sv1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      sv0 = std::fabs(sv0);
    if (inst_.neg & (1u << 0))
      sv0 = -sv0;
    if (inst_.abs & (1u << 1))
      sv1 = std::fabs(sv1);
    if (inst_.neg & (1u << 1))
      sv1 = -sv1;
    double result = sv0 + sv1;
    if (inst_.omod == 1)
      result *= 2.0;
    else if (inst_.omod == 2)
      result *= 4.0;
    else if (inst_.omod == 3)
      result *= 0.5;
    if (inst_.clamp)
      result = std::clamp(result, 0.0, 1.0);
    vdst.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));
  }
}

VMulF64Vop3::VMulF64Vop3(const MachineInst *inst)
    : Vop3("v_mul_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMulF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double sv0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double sv1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      sv0 = std::fabs(sv0);
    if (inst_.neg & (1u << 0))
      sv0 = -sv0;
    if (inst_.abs & (1u << 1))
      sv1 = std::fabs(sv1);
    if (inst_.neg & (1u << 1))
      sv1 = -sv1;
    double result = sv0 * sv1;
    if (inst_.omod == 1)
      result *= 2.0;
    else if (inst_.omod == 2)
      result *= 4.0;
    else if (inst_.omod == 3)
      result *= 0.5;
    if (inst_.clamp)
      result = std::clamp(result, 0.0, 1.0);
    vdst.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));
  }
}

VMinF64Vop3::VMinF64Vop3(const MachineInst *inst)
    : Vop3("v_min_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMinF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double sv0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double sv1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      sv0 = std::fabs(sv0);
    if (inst_.neg & (1u << 0))
      sv0 = -sv0;
    if (inst_.abs & (1u << 1))
      sv1 = std::fabs(sv1);
    if (inst_.neg & (1u << 1))
      sv1 = -sv1;
    double result = std::fmin(sv0, sv1);
    if (inst_.omod == 1)
      result *= 2.0;
    else if (inst_.omod == 2)
      result *= 4.0;
    else if (inst_.omod == 3)
      result *= 0.5;
    if (inst_.clamp)
      result = std::clamp(result, 0.0, 1.0);
    vdst.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));
  }
}

VMaxF64Vop3::VMaxF64Vop3(const MachineInst *inst)
    : Vop3("v_max_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMaxF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double sv0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double sv1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      sv0 = std::fabs(sv0);
    if (inst_.neg & (1u << 0))
      sv0 = -sv0;
    if (inst_.abs & (1u << 1))
      sv1 = std::fabs(sv1);
    if (inst_.neg & (1u << 1))
      sv1 = -sv1;
    double result = std::fmax(sv0, sv1);
    if (inst_.omod == 1)
      result *= 2.0;
    else if (inst_.omod == 2)
      result *= 4.0;
    else if (inst_.omod == 3)
      result *= 0.5;
    if (inst_.clamp)
      result = std::clamp(result, 0.0, 1.0);
    vdst.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));
  }
}

VLdexpF64Vop3::VLdexpF64Vop3(const MachineInst *inst)
    : Vop3("v_ldexp_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VLdexpF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double sv0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    int32_t sv1_i = static_cast<int32_t>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      sv0 = std::fabs(sv0);
    if (inst_.neg & (1u << 0))
      sv0 = -sv0;
    double result = std::ldexp(sv0, static_cast<int>(sv1_i));
    if (inst_.omod == 1)
      result *= 2.0;
    else if (inst_.omod == 2)
      result *= 4.0;
    else if (inst_.omod == 3)
      result *= 0.5;
    if (inst_.clamp)
      result = std::clamp(result, 0.0, 1.0);
    vdst.write_lane64(wf, lane, std::bit_cast<uint64_t>(result));
  }
}

VMulLoU32Vop3::VMulLoU32Vop3(const MachineInst *inst)
    : Vop3("v_mul_lo_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMulLoU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = src1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, sv0 * sv1);
  }
}

VMulHiU32Vop3::VMulHiU32Vop3(const MachineInst *inst)
    : Vop3("v_mul_hi_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMulHiU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t sv0 = src0.read_lane(wf, lane);
    uint32_t sv1 = src1.read_lane(wf, lane);
    vdst.write_lane(wf, lane, static_cast<uint32_t>((static_cast<uint64_t>(sv0) * sv1) >> 32));
  }
}

VMulHiI32Vop3::VMulHiI32Vop3(const MachineInst *inst)
    : Vop3("v_mul_hi_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VMulHiI32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t sv0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t sv1 = static_cast<int32_t>(src1.read_lane(wf, lane));
    vdst.write_lane(
        wf, lane,
        static_cast<uint32_t>(static_cast<uint64_t>(static_cast<int64_t>(sv0) * sv1) >> 32));
  }
}

VTrigPreopF64Vop3::VTrigPreopF64Vop3(const MachineInst *inst)
    : Vop3("v_trig_preop_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VTrigPreopF64Vop3::execute(amdgpu::Wavefront &wf) { (void)wf; }

VLshlrevB16Vop3::VLshlrevB16Vop3(const MachineInst *inst)
    : Vop3("v_lshlrev_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VLshlrevB16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t sv0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t sv1 = static_cast<uint16_t>(src1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(static_cast<uint16_t>(sv1 << (sv0 & 15u))));
  }
}

VLshrrevB16Vop3::VLshrrevB16Vop3(const MachineInst *inst)
    : Vop3("v_lshrrev_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VLshrrevB16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t sv0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t sv1 = static_cast<uint16_t>(src1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(static_cast<uint16_t>(sv1 >> (sv0 & 15u))));
  }
}

VAshrrevI16Vop3::VAshrrevI16Vop3(const MachineInst *inst)
    : Vop3("v_ashrrev_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VAshrrevI16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t sv0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t sv1 = static_cast<int16_t>(src1.read_lane(wf, lane) & 0xFFFF);
    vdst.write_lane(
        wf, lane,
        static_cast<uint32_t>(static_cast<uint16_t>(static_cast<int16_t>(sv1 >> (sv0 & 15)))));
  }
}

VLshlrevB64Vop3::VLshlrevB64Vop3(const MachineInst *inst)
    : Vop3("v_lshlrev_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VLshlrevB64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t val = src1.read_lane64(wf, lane);
    uint32_t shift = src0.read_lane(wf, lane) & 63u;
    vdst.write_lane64(wf, lane, val << shift);
  }
}

VLshrrevB64Vop3::VLshrrevB64Vop3(const MachineInst *inst)
    : Vop3("v_lshrrev_b64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VLshrrevB64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t val = src1.read_lane64(wf, lane);
    uint32_t shift = src0.read_lane(wf, lane) & 63u;
    vdst.write_lane64(wf, lane, val >> shift);
  }
}

VAshrrevI64Vop3::VAshrrevI64Vop3(const MachineInst *inst)
    : Vop3("v_ashrrev_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VAshrrevI64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int64_t val = static_cast<int64_t>(src1.read_lane64(wf, lane));
    uint32_t shift = src0.read_lane(wf, lane) & 63u;
    vdst.write_lane64(wf, lane, static_cast<uint64_t>(val >> shift));
  }
}

VReadlaneB32Vop3::VReadlaneB32Vop3(const MachineInst *inst)
    : Vop3("v_readlane_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SSRC_LANESEL, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VReadlaneB32Vop3::execute(amdgpu::Wavefront &wf) {
  uint32_t lane = src1.read_scalar(wf);
  vdst.write_scalar(wf, src0.read_lane(wf, lane));
}

VWritelaneB32Vop3::VWritelaneB32Vop3(const MachineInst *inst)
    : Vop3("v_writelane_b32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SSRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SSRC_LANESEL, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  src_operands_.emplace_back(&vdst);
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VWritelaneB32Vop3::execute(amdgpu::Wavefront &wf) {
  uint32_t val = src0.read_scalar(wf);
  uint32_t lane = src1.read_scalar(wf);
  vdst.write_lane(wf, lane, val);
}

VAndB16Vop3::VAndB16Vop3(const MachineInst *inst)
    : Vop3("v_and_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VAndB16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t sv0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t sv1 = static_cast<uint16_t>(src1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(static_cast<uint16_t>(sv0 & sv1)));
  }
}

VOrB16Vop3::VOrB16Vop3(const MachineInst *inst)
    : Vop3("v_or_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VOrB16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t sv0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t sv1 = static_cast<uint16_t>(src1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(static_cast<uint16_t>(sv0 | sv1)));
  }
}

VXorB16Vop3::VXorB16Vop3(const MachineInst *inst)
    : Vop3("v_xor_b16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VXorB16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t sv0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t sv1 = static_cast<uint16_t>(src1.read_lane(wf, lane));
    vdst.write_lane(wf, lane, static_cast<uint32_t>(static_cast<uint16_t>(sv0 ^ sv1)));
  }
}

VCmpFF16Vop3::VCmpFF16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_f_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpFF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLtF16Vop3::VCmpLtF16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_lt_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpLtF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 < s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpEqF16Vop3::VCmpEqF16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_eq_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpEqF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 == s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLeF16Vop3::VCmpLeF16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_le_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpLeF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 <= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGtF16Vop3::VCmpGtF16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_gt_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpGtF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 > s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLgF16Vop3::VCmpLgF16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_lg_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpLgF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 < s1 || s0 > s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGeF16Vop3::VCmpGeF16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_ge_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpGeF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 >= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpOF16Vop3::VCmpOF16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_o_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpOF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!std::isnan(s0) && !std::isnan(s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpUF16Vop3::VCmpUF16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_u_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpUF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (std::isnan(s0) || std::isnan(s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNgeF16Vop3::VCmpNgeF16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_nge_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpNgeF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 >= s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNlgF16Vop3::VCmpNlgF16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_nlg_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpNlgF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 < s1 || s0 > s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNgtF16Vop3::VCmpNgtF16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_ngt_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpNgtF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 > s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNleF16Vop3::VCmpNleF16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_nle_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpNleF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 <= s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNeqF16Vop3::VCmpNeqF16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_neq_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpNeqF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 != s1 || std::isnan(s0) || std::isnan(s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNltF16Vop3::VCmpNltF16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_nlt_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpNltF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 < s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpTF16Vop3::VCmpTF16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_t_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpTF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc |= (1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpFF32Vop3::VCmpFF32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_f_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpFF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLtF32Vop3::VCmpLtF32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_lt_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpLtF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 < s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpEqF32Vop3::VCmpEqF32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_eq_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpEqF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 == s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLeF32Vop3::VCmpLeF32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_le_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpLeF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 <= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGtF32Vop3::VCmpGtF32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_gt_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpGtF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 > s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLgF32Vop3::VCmpLgF32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_lg_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpLgF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 < s1 || s0 > s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGeF32Vop3::VCmpGeF32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_ge_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpGeF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 >= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpOF32Vop3::VCmpOF32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_o_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpOF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!std::isnan(s0) && !std::isnan(s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpUF32Vop3::VCmpUF32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_u_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpUF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (std::isnan(s0) || std::isnan(s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNgeF32Vop3::VCmpNgeF32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_nge_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpNgeF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 >= s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNlgF32Vop3::VCmpNlgF32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_nlg_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpNlgF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 < s1 || s0 > s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNgtF32Vop3::VCmpNgtF32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_ngt_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpNgtF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 > s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNleF32Vop3::VCmpNleF32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_nle_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpNleF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 <= s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNeqF32Vop3::VCmpNeqF32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_neq_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpNeqF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 != s1 || std::isnan(s0) || std::isnan(s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNltF32Vop3::VCmpNltF32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_nlt_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpNltF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 < s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpTF32Vop3::VCmpTF32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_t_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpTF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc |= (1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpFF64Vop3::VCmpFF64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_f_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpFF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLtF64Vop3::VCmpLtF64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_lt_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpLtF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 < s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpEqF64Vop3::VCmpEqF64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_eq_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpEqF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 == s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLeF64Vop3::VCmpLeF64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_le_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpLeF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 <= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGtF64Vop3::VCmpGtF64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_gt_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpGtF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 > s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLgF64Vop3::VCmpLgF64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_lg_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpLgF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 < s1 || s0 > s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGeF64Vop3::VCmpGeF64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_ge_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpGeF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 >= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpOF64Vop3::VCmpOF64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_o_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpOF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!std::isnan(s0) && !std::isnan(s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpUF64Vop3::VCmpUF64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_u_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpUF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (std::isnan(s0) || std::isnan(s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNgeF64Vop3::VCmpNgeF64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_nge_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpNgeF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 >= s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNlgF64Vop3::VCmpNlgF64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_nlg_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpNlgF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 < s1 || s0 > s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNgtF64Vop3::VCmpNgtF64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_ngt_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpNgtF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 > s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNleF64Vop3::VCmpNleF64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_nle_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpNleF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 <= s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNeqF64Vop3::VCmpNeqF64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_neq_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpNeqF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 != s1 || std::isnan(s0) || std::isnan(s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNltF64Vop3::VCmpNltF64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_nlt_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpNltF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 < s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpTF64Vop3::VCmpTF64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_t_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpTF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc |= (1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLtI16Vop3::VCmpLtI16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_lt_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpLtI16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t s0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t s1 = static_cast<int16_t>(src1.read_lane(wf, lane) & 0xFFFF);
    if (s0 < s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpEqI16Vop3::VCmpEqI16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_eq_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpEqI16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t s0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t s1 = static_cast<int16_t>(src1.read_lane(wf, lane) & 0xFFFF);
    if (s0 == s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLeI16Vop3::VCmpLeI16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_le_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpLeI16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t s0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t s1 = static_cast<int16_t>(src1.read_lane(wf, lane) & 0xFFFF);
    if (s0 <= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGtI16Vop3::VCmpGtI16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_gt_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpGtI16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t s0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t s1 = static_cast<int16_t>(src1.read_lane(wf, lane) & 0xFFFF);
    if (s0 > s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNeI16Vop3::VCmpNeI16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_ne_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpNeI16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t s0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t s1 = static_cast<int16_t>(src1.read_lane(wf, lane) & 0xFFFF);
    if (s0 != s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGeI16Vop3::VCmpGeI16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_ge_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpGeI16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t s0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t s1 = static_cast<int16_t>(src1.read_lane(wf, lane) & 0xFFFF);
    if (s0 >= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLtU16Vop3::VCmpLtU16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_lt_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpLtU16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t s1 = static_cast<uint16_t>(src1.read_lane(wf, lane));
    if (s0 < s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpEqU16Vop3::VCmpEqU16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_eq_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpEqU16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t s1 = static_cast<uint16_t>(src1.read_lane(wf, lane));
    if (s0 == s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLeU16Vop3::VCmpLeU16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_le_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpLeU16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t s1 = static_cast<uint16_t>(src1.read_lane(wf, lane));
    if (s0 <= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGtU16Vop3::VCmpGtU16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_gt_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpGtU16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t s1 = static_cast<uint16_t>(src1.read_lane(wf, lane));
    if (s0 > s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNeU16Vop3::VCmpNeU16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_ne_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpNeU16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t s1 = static_cast<uint16_t>(src1.read_lane(wf, lane));
    if (s0 != s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGeU16Vop3::VCmpGeU16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_ge_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpGeU16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t s1 = static_cast<uint16_t>(src1.read_lane(wf, lane));
    if (s0 >= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpFI32Vop3::VCmpFI32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_f_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpFI32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLtI32Vop3::VCmpLtI32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_lt_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpLtI32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t s1 = static_cast<int32_t>(src1.read_lane(wf, lane));
    if (s0 < s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpEqI32Vop3::VCmpEqI32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_eq_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpEqI32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t s1 = static_cast<int32_t>(src1.read_lane(wf, lane));
    if (s0 == s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLeI32Vop3::VCmpLeI32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_le_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpLeI32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t s1 = static_cast<int32_t>(src1.read_lane(wf, lane));
    if (s0 <= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGtI32Vop3::VCmpGtI32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_gt_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpGtI32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t s1 = static_cast<int32_t>(src1.read_lane(wf, lane));
    if (s0 > s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNeI32Vop3::VCmpNeI32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_ne_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpNeI32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t s1 = static_cast<int32_t>(src1.read_lane(wf, lane));
    if (s0 != s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGeI32Vop3::VCmpGeI32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_ge_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpGeI32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t s1 = static_cast<int32_t>(src1.read_lane(wf, lane));
    if (s0 >= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpTI32Vop3::VCmpTI32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_t_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpTI32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc |= (1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpFU32Vop3::VCmpFU32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_f_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpFU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLtU32Vop3::VCmpLtU32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_lt_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpLtU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane);
    uint32_t s1 = src1.read_lane(wf, lane);
    if (s0 < s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpEqU32Vop3::VCmpEqU32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_eq_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpEqU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane);
    uint32_t s1 = src1.read_lane(wf, lane);
    if (s0 == s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLeU32Vop3::VCmpLeU32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_le_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpLeU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane);
    uint32_t s1 = src1.read_lane(wf, lane);
    if (s0 <= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGtU32Vop3::VCmpGtU32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_gt_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpGtU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane);
    uint32_t s1 = src1.read_lane(wf, lane);
    if (s0 > s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNeU32Vop3::VCmpNeU32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_ne_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpNeU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane);
    uint32_t s1 = src1.read_lane(wf, lane);
    if (s0 != s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGeU32Vop3::VCmpGeU32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_ge_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpGeU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane);
    uint32_t s1 = src1.read_lane(wf, lane);
    if (s0 >= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpTU32Vop3::VCmpTU32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_t_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpTU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc |= (1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpFI64Vop3::VCmpFI64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_f_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpFI64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLtI64Vop3::VCmpLtI64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_lt_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpLtI64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int64_t s0 = static_cast<int64_t>(src0.read_lane64(wf, lane));
    int64_t s1 = static_cast<int64_t>(src1.read_lane64(wf, lane));
    if (s0 < s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpEqI64Vop3::VCmpEqI64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_eq_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpEqI64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int64_t s0 = static_cast<int64_t>(src0.read_lane64(wf, lane));
    int64_t s1 = static_cast<int64_t>(src1.read_lane64(wf, lane));
    if (s0 == s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLeI64Vop3::VCmpLeI64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_le_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpLeI64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int64_t s0 = static_cast<int64_t>(src0.read_lane64(wf, lane));
    int64_t s1 = static_cast<int64_t>(src1.read_lane64(wf, lane));
    if (s0 <= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGtI64Vop3::VCmpGtI64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_gt_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpGtI64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int64_t s0 = static_cast<int64_t>(src0.read_lane64(wf, lane));
    int64_t s1 = static_cast<int64_t>(src1.read_lane64(wf, lane));
    if (s0 > s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNeI64Vop3::VCmpNeI64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_ne_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpNeI64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int64_t s0 = static_cast<int64_t>(src0.read_lane64(wf, lane));
    int64_t s1 = static_cast<int64_t>(src1.read_lane64(wf, lane));
    if (s0 != s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGeI64Vop3::VCmpGeI64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_ge_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpGeI64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int64_t s0 = static_cast<int64_t>(src0.read_lane64(wf, lane));
    int64_t s1 = static_cast<int64_t>(src1.read_lane64(wf, lane));
    if (s0 >= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpTI64Vop3::VCmpTI64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_t_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpTI64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc |= (1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpFU64Vop3::VCmpFU64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_f_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpFU64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLtU64Vop3::VCmpLtU64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_lt_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpLtU64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t s0 = src0.read_lane64(wf, lane);
    uint64_t s1 = src1.read_lane64(wf, lane);
    if (s0 < s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpEqU64Vop3::VCmpEqU64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_eq_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpEqU64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t s0 = src0.read_lane64(wf, lane);
    uint64_t s1 = src1.read_lane64(wf, lane);
    if (s0 == s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLeU64Vop3::VCmpLeU64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_le_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpLeU64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t s0 = src0.read_lane64(wf, lane);
    uint64_t s1 = src1.read_lane64(wf, lane);
    if (s0 <= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGtU64Vop3::VCmpGtU64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_gt_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpGtU64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t s0 = src0.read_lane64(wf, lane);
    uint64_t s1 = src1.read_lane64(wf, lane);
    if (s0 > s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNeU64Vop3::VCmpNeU64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_ne_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpNeU64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t s0 = src0.read_lane64(wf, lane);
    uint64_t s1 = src1.read_lane64(wf, lane);
    if (s0 != s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGeU64Vop3::VCmpGeU64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_ge_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpGeU64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t s0 = src0.read_lane64(wf, lane);
    uint64_t s1 = src1.read_lane64(wf, lane);
    if (s0 >= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpTU64Vop3::VCmpTU64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_t_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpTU64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc |= (1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpClassF16Vop3::VCmpClassF16Vop3(const MachineInst *inst)
    : Vop3("v_cmp_class_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpClassF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0_raw = static_cast<uint16_t>(src0.read_lane(wf, lane));
    float s0 = util::f16_to_f32(s0_raw);
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    uint32_t mask = src1.read_lane(wf, lane);
    bool match = false;
    bool is_f16_nan = ((s0_raw & 0x7C00) == 0x7C00) && ((s0_raw & 0x03FF) != 0);
    if ((mask & 0x001) && is_f16_nan && (s0_raw & 0x0200) == 0)
      match = true;
    if ((mask & 0x002) && is_f16_nan && (s0_raw & 0x0200) != 0)
      match = true;
    if ((mask & 0x004) && std::isinf(s0) && s0 < 0)
      match = true;
    if ((mask & 0x008) && std::isnormal(s0) && s0 < 0)
      match = true;
    if ((mask & 0x010) && !std::isnormal(s0) && !std::isinf(s0) && !std::isnan(s0) && s0 != 0.0f &&
        std::signbit(s0))
      match = true;
    if ((mask & 0x020) && s0 == 0.0f && std::signbit(s0))
      match = true;
    if ((mask & 0x040) && s0 == 0.0f && !std::signbit(s0))
      match = true;
    if ((mask & 0x080) && !std::isnormal(s0) && !std::isinf(s0) && !std::isnan(s0) && s0 != 0.0f &&
        !std::signbit(s0))
      match = true;
    if ((mask & 0x100) && std::isnormal(s0) && s0 > 0)
      match = true;
    if ((mask & 0x200) && std::isinf(s0) && s0 > 0)
      match = true;
    if (match)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpClassF32Vop3::VCmpClassF32Vop3(const MachineInst *inst)
    : Vop3("v_cmp_class_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpClassF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    uint32_t mask = src1.read_lane(wf, lane);
    bool match = false;
    if ((mask & 0x001) && std::isnan(s0) && (std::bit_cast<uint32_t>(s0) & 0x00400000) == 0)
      match = true;
    if ((mask & 0x002) && std::isnan(s0) && (std::bit_cast<uint32_t>(s0) & 0x00400000) != 0)
      match = true;
    if ((mask & 0x004) && std::isinf(s0) && s0 < 0)
      match = true;
    if ((mask & 0x008) && std::isnormal(s0) && s0 < 0)
      match = true;
    if ((mask & 0x010) && !std::isnormal(s0) && !std::isinf(s0) && !std::isnan(s0) && s0 != 0.0f &&
        std::signbit(s0))
      match = true;
    if ((mask & 0x020) && s0 == 0.0f && std::signbit(s0))
      match = true;
    if ((mask & 0x040) && s0 == 0.0f && !std::signbit(s0))
      match = true;
    if ((mask & 0x080) && !std::isnormal(s0) && !std::isinf(s0) && !std::isnan(s0) && s0 != 0.0f &&
        !std::signbit(s0))
      match = true;
    if ((mask & 0x100) && std::isnormal(s0) && s0 > 0)
      match = true;
    if ((mask & 0x200) && std::isinf(s0) && s0 > 0)
      match = true;
    if (match)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpClassF64Vop3::VCmpClassF64Vop3(const MachineInst *inst)
    : Vop3("v_cmp_class_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_SREG, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpClassF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    uint32_t mask = src1.read_lane(wf, lane);
    bool match = false;
    if ((mask & 0x001) && std::isnan(s0) &&
        (std::bit_cast<uint64_t>(s0) & 0x0008000000000000ULL) == 0)
      match = true;
    if ((mask & 0x002) && std::isnan(s0) &&
        (std::bit_cast<uint64_t>(s0) & 0x0008000000000000ULL) != 0)
      match = true;
    if ((mask & 0x004) && std::isinf(s0) && s0 < 0)
      match = true;
    if ((mask & 0x008) && std::isnormal(s0) && s0 < 0)
      match = true;
    if ((mask & 0x010) && !std::isnormal(s0) && !std::isinf(s0) && !std::isnan(s0) && s0 != 0.0 &&
        std::signbit(s0))
      match = true;
    if ((mask & 0x020) && s0 == 0.0 && std::signbit(s0))
      match = true;
    if ((mask & 0x040) && s0 == 0.0 && !std::signbit(s0))
      match = true;
    if ((mask & 0x080) && !std::isnormal(s0) && !std::isinf(s0) && !std::isnan(s0) && s0 != 0.0 &&
        !std::signbit(s0))
      match = true;
    if ((mask & 0x100) && std::isnormal(s0) && s0 > 0)
      match = true;
    if ((mask & 0x200) && std::isinf(s0) && s0 > 0)
      match = true;
    if (match)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpxFF16Vop3::VCmpxFF16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_f_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxFF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    (void)lane;
  }
  wf.set_exec(result);
}

VCmpxLtF16Vop3::VCmpxLtF16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_lt_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxLtF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 < s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxEqF16Vop3::VCmpxEqF16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_eq_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxEqF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 == s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxLeF16Vop3::VCmpxLeF16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_le_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxLeF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 <= s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxGtF16Vop3::VCmpxGtF16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_gt_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxGtF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 > s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxLgF16Vop3::VCmpxLgF16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_lg_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxLgF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 < s1 || s0 > s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxGeF16Vop3::VCmpxGeF16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_ge_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxGeF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 >= s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxOF16Vop3::VCmpxOF16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_o_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxOF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!std::isnan(s0) && !std::isnan(s1))
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxUF16Vop3::VCmpxUF16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_u_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxUF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (std::isnan(s0) || std::isnan(s1))
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxNgeF16Vop3::VCmpxNgeF16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_nge_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxNgeF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 >= s1))
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxNlgF16Vop3::VCmpxNlgF16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_nlg_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxNlgF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 < s1 || s0 > s1))
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxNgtF16Vop3::VCmpxNgtF16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_ngt_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxNgtF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 > s1))
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxNleF16Vop3::VCmpxNleF16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_nle_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxNleF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 <= s1))
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxNeqF16Vop3::VCmpxNeqF16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_neq_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxNeqF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 != s1 || std::isnan(s0) || std::isnan(s1))
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxNltF16Vop3::VCmpxNltF16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_nlt_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxNltF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(src1.read_lane(wf, lane)));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 < s1))
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxTF16Vop3::VCmpxTF16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_t_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxTF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxFF32Vop3::VCmpxFF32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_f_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxFF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    (void)lane;
  }
  wf.set_exec(result);
}

VCmpxLtF32Vop3::VCmpxLtF32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_lt_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxLtF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 < s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxEqF32Vop3::VCmpxEqF32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_eq_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxEqF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 == s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxLeF32Vop3::VCmpxLeF32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_le_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxLeF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 <= s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxGtF32Vop3::VCmpxGtF32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_gt_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxGtF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 > s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxLgF32Vop3::VCmpxLgF32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_lg_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxLgF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 < s1 || s0 > s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxGeF32Vop3::VCmpxGeF32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_ge_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxGeF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 >= s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxOF32Vop3::VCmpxOF32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_o_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxOF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!std::isnan(s0) && !std::isnan(s1))
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxUF32Vop3::VCmpxUF32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_u_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxUF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (std::isnan(s0) || std::isnan(s1))
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxNgeF32Vop3::VCmpxNgeF32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_nge_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxNgeF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 >= s1))
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxNlgF32Vop3::VCmpxNlgF32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_nlg_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxNlgF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 < s1 || s0 > s1))
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxNgtF32Vop3::VCmpxNgtF32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_ngt_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxNgtF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 > s1))
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxNleF32Vop3::VCmpxNleF32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_nle_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxNleF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 <= s1))
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxNeqF32Vop3::VCmpxNeqF32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_neq_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxNeqF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 != s1 || std::isnan(s0) || std::isnan(s1))
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxNltF32Vop3::VCmpxNltF32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_nlt_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxNltF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(src1.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 < s1))
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxTF32Vop3::VCmpxTF32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_t_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxTF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxFF64Vop3::VCmpxFF64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_f_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxFF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    (void)lane;
  }
  wf.set_exec(result);
}

VCmpxLtF64Vop3::VCmpxLtF64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_lt_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxLtF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 < s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxEqF64Vop3::VCmpxEqF64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_eq_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxEqF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 == s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxLeF64Vop3::VCmpxLeF64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_le_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxLeF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 <= s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxGtF64Vop3::VCmpxGtF64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_gt_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxGtF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 > s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxLgF64Vop3::VCmpxLgF64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_lg_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxLgF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 < s1 || s0 > s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxGeF64Vop3::VCmpxGeF64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_ge_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxGeF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 >= s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxOF64Vop3::VCmpxOF64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_o_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxOF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!std::isnan(s0) && !std::isnan(s1))
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxUF64Vop3::VCmpxUF64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_u_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxUF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (std::isnan(s0) || std::isnan(s1))
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxNgeF64Vop3::VCmpxNgeF64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_nge_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxNgeF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 >= s1))
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxNlgF64Vop3::VCmpxNlgF64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_nlg_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxNlgF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 < s1 || s0 > s1))
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxNgtF64Vop3::VCmpxNgtF64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_ngt_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxNgtF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 > s1))
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxNleF64Vop3::VCmpxNleF64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_nle_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxNleF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 <= s1))
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxNeqF64Vop3::VCmpxNeqF64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_neq_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxNeqF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (s0 != s1 || std::isnan(s0) || std::isnan(s1))
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxNltF64Vop3::VCmpxNltF64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_nlt_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxNltF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(src1.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    if (inst_.abs & (1u << 1))
      s1 = std::fabs(s1);
    if (inst_.neg & (1u << 1))
      s1 = -s1;
    if (!(s0 < s1))
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxTF64Vop3::VCmpxTF64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_t_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxTF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxLtI16Vop3::VCmpxLtI16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_lt_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxLtI16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t s0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t s1 = static_cast<int16_t>(src1.read_lane(wf, lane) & 0xFFFF);
    if (s0 < s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxEqI16Vop3::VCmpxEqI16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_eq_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxEqI16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t s0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t s1 = static_cast<int16_t>(src1.read_lane(wf, lane) & 0xFFFF);
    if (s0 == s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxLeI16Vop3::VCmpxLeI16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_le_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxLeI16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t s0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t s1 = static_cast<int16_t>(src1.read_lane(wf, lane) & 0xFFFF);
    if (s0 <= s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxGtI16Vop3::VCmpxGtI16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_gt_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxGtI16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t s0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t s1 = static_cast<int16_t>(src1.read_lane(wf, lane) & 0xFFFF);
    if (s0 > s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxNeI16Vop3::VCmpxNeI16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_ne_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxNeI16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t s0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t s1 = static_cast<int16_t>(src1.read_lane(wf, lane) & 0xFFFF);
    if (s0 != s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxGeI16Vop3::VCmpxGeI16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_ge_i16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxGeI16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t s0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t s1 = static_cast<int16_t>(src1.read_lane(wf, lane) & 0xFFFF);
    if (s0 >= s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxLtU16Vop3::VCmpxLtU16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_lt_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxLtU16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t s1 = static_cast<uint16_t>(src1.read_lane(wf, lane));
    if (s0 < s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxEqU16Vop3::VCmpxEqU16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_eq_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxEqU16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t s1 = static_cast<uint16_t>(src1.read_lane(wf, lane));
    if (s0 == s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxLeU16Vop3::VCmpxLeU16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_le_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxLeU16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t s1 = static_cast<uint16_t>(src1.read_lane(wf, lane));
    if (s0 <= s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxGtU16Vop3::VCmpxGtU16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_gt_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxGtU16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t s1 = static_cast<uint16_t>(src1.read_lane(wf, lane));
    if (s0 > s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxNeU16Vop3::VCmpxNeU16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_ne_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxNeU16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t s1 = static_cast<uint16_t>(src1.read_lane(wf, lane));
    if (s0 != s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxGeU16Vop3::VCmpxGeU16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_ge_u16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxGeU16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t s1 = static_cast<uint16_t>(src1.read_lane(wf, lane));
    if (s0 >= s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxFI32Vop3::VCmpxFI32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_f_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxFI32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    (void)lane;
  }
  wf.set_exec(result);
}

VCmpxLtI32Vop3::VCmpxLtI32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_lt_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxLtI32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t s1 = static_cast<int32_t>(src1.read_lane(wf, lane));
    if (s0 < s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxEqI32Vop3::VCmpxEqI32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_eq_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxEqI32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t s1 = static_cast<int32_t>(src1.read_lane(wf, lane));
    if (s0 == s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxLeI32Vop3::VCmpxLeI32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_le_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxLeI32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t s1 = static_cast<int32_t>(src1.read_lane(wf, lane));
    if (s0 <= s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxGtI32Vop3::VCmpxGtI32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_gt_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxGtI32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t s1 = static_cast<int32_t>(src1.read_lane(wf, lane));
    if (s0 > s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxNeI32Vop3::VCmpxNeI32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_ne_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxNeI32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t s1 = static_cast<int32_t>(src1.read_lane(wf, lane));
    if (s0 != s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxGeI32Vop3::VCmpxGeI32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_ge_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxGeI32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t s1 = static_cast<int32_t>(src1.read_lane(wf, lane));
    if (s0 >= s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxTI32Vop3::VCmpxTI32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_t_i32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxTI32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxFU32Vop3::VCmpxFU32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_f_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxFU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    (void)lane;
  }
  wf.set_exec(result);
}

VCmpxLtU32Vop3::VCmpxLtU32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_lt_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxLtU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane);
    uint32_t s1 = src1.read_lane(wf, lane);
    if (s0 < s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxEqU32Vop3::VCmpxEqU32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_eq_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxEqU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane);
    uint32_t s1 = src1.read_lane(wf, lane);
    if (s0 == s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxLeU32Vop3::VCmpxLeU32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_le_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxLeU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane);
    uint32_t s1 = src1.read_lane(wf, lane);
    if (s0 <= s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxGtU32Vop3::VCmpxGtU32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_gt_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxGtU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane);
    uint32_t s1 = src1.read_lane(wf, lane);
    if (s0 > s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxNeU32Vop3::VCmpxNeU32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_ne_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxNeU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane);
    uint32_t s1 = src1.read_lane(wf, lane);
    if (s0 != s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxGeU32Vop3::VCmpxGeU32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_ge_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxGeU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane);
    uint32_t s1 = src1.read_lane(wf, lane);
    if (s0 >= s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxTU32Vop3::VCmpxTU32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_t_u32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxTU32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxFI64Vop3::VCmpxFI64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_f_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxFI64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    (void)lane;
  }
  wf.set_exec(result);
}

VCmpxLtI64Vop3::VCmpxLtI64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_lt_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxLtI64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int64_t s0 = static_cast<int64_t>(src0.read_lane64(wf, lane));
    int64_t s1 = static_cast<int64_t>(src1.read_lane64(wf, lane));
    if (s0 < s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxEqI64Vop3::VCmpxEqI64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_eq_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxEqI64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int64_t s0 = static_cast<int64_t>(src0.read_lane64(wf, lane));
    int64_t s1 = static_cast<int64_t>(src1.read_lane64(wf, lane));
    if (s0 == s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxLeI64Vop3::VCmpxLeI64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_le_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxLeI64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int64_t s0 = static_cast<int64_t>(src0.read_lane64(wf, lane));
    int64_t s1 = static_cast<int64_t>(src1.read_lane64(wf, lane));
    if (s0 <= s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxGtI64Vop3::VCmpxGtI64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_gt_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxGtI64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int64_t s0 = static_cast<int64_t>(src0.read_lane64(wf, lane));
    int64_t s1 = static_cast<int64_t>(src1.read_lane64(wf, lane));
    if (s0 > s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxNeI64Vop3::VCmpxNeI64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_ne_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxNeI64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int64_t s0 = static_cast<int64_t>(src0.read_lane64(wf, lane));
    int64_t s1 = static_cast<int64_t>(src1.read_lane64(wf, lane));
    if (s0 != s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxGeI64Vop3::VCmpxGeI64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_ge_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxGeI64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int64_t s0 = static_cast<int64_t>(src0.read_lane64(wf, lane));
    int64_t s1 = static_cast<int64_t>(src1.read_lane64(wf, lane));
    if (s0 >= s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxTI64Vop3::VCmpxTI64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_t_i64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxTI64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxFU64Vop3::VCmpxFU64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_f_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxFU64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    (void)lane;
  }
  wf.set_exec(result);
}

VCmpxLtU64Vop3::VCmpxLtU64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_lt_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxLtU64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t s0 = src0.read_lane64(wf, lane);
    uint64_t s1 = src1.read_lane64(wf, lane);
    if (s0 < s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxEqU64Vop3::VCmpxEqU64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_eq_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxEqU64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t s0 = src0.read_lane64(wf, lane);
    uint64_t s1 = src1.read_lane64(wf, lane);
    if (s0 == s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxLeU64Vop3::VCmpxLeU64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_le_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxLeU64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t s0 = src0.read_lane64(wf, lane);
    uint64_t s1 = src1.read_lane64(wf, lane);
    if (s0 <= s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxGtU64Vop3::VCmpxGtU64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_gt_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxGtU64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t s0 = src0.read_lane64(wf, lane);
    uint64_t s1 = src1.read_lane64(wf, lane);
    if (s0 > s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxNeU64Vop3::VCmpxNeU64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_ne_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxNeU64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t s0 = src0.read_lane64(wf, lane);
    uint64_t s1 = src1.read_lane64(wf, lane);
    if (s0 != s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxGeU64Vop3::VCmpxGeU64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_ge_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxGeU64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t s0 = src0.read_lane64(wf, lane);
    uint64_t s1 = src1.read_lane64(wf, lane);
    if (s0 >= s1)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxTU64Vop3::VCmpxTU64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_t_u64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxTU64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxClassF16Vop3::VCmpxClassF16Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_class_f16", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxClassF16Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0_raw = static_cast<uint16_t>(src0.read_lane(wf, lane));
    float s0 = util::f16_to_f32(s0_raw);
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    uint32_t mask = src1.read_lane(wf, lane);
    bool match = false;
    bool is_f16_nan = ((s0_raw & 0x7C00) == 0x7C00) && ((s0_raw & 0x03FF) != 0);
    if ((mask & 0x001) && is_f16_nan && (s0_raw & 0x0200) == 0)
      match = true;
    if ((mask & 0x002) && is_f16_nan && (s0_raw & 0x0200) != 0)
      match = true;
    if ((mask & 0x004) && std::isinf(s0) && s0 < 0)
      match = true;
    if ((mask & 0x008) && std::isnormal(s0) && s0 < 0)
      match = true;
    if ((mask & 0x010) && !std::isnormal(s0) && !std::isinf(s0) && !std::isnan(s0) && s0 != 0.0f &&
        std::signbit(s0))
      match = true;
    if ((mask & 0x020) && s0 == 0.0f && std::signbit(s0))
      match = true;
    if ((mask & 0x040) && s0 == 0.0f && !std::signbit(s0))
      match = true;
    if ((mask & 0x080) && !std::isnormal(s0) && !std::isinf(s0) && !std::isnan(s0) && s0 != 0.0f &&
        !std::signbit(s0))
      match = true;
    if ((mask & 0x100) && std::isnormal(s0) && s0 > 0)
      match = true;
    if ((mask & 0x200) && std::isinf(s0) && s0 > 0)
      match = true;
    if (match)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxClassF32Vop3::VCmpxClassF32Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_class_f32", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxClassF32Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    uint32_t mask = src1.read_lane(wf, lane);
    bool match = false;
    if ((mask & 0x001) && std::isnan(s0) && (std::bit_cast<uint32_t>(s0) & 0x00400000) == 0)
      match = true;
    if ((mask & 0x002) && std::isnan(s0) && (std::bit_cast<uint32_t>(s0) & 0x00400000) != 0)
      match = true;
    if ((mask & 0x004) && std::isinf(s0) && s0 < 0)
      match = true;
    if ((mask & 0x008) && std::isnormal(s0) && s0 < 0)
      match = true;
    if ((mask & 0x010) && !std::isnormal(s0) && !std::isinf(s0) && !std::isnan(s0) && s0 != 0.0f &&
        std::signbit(s0))
      match = true;
    if ((mask & 0x020) && s0 == 0.0f && std::signbit(s0))
      match = true;
    if ((mask & 0x040) && s0 == 0.0f && !std::signbit(s0))
      match = true;
    if ((mask & 0x080) && !std::isnormal(s0) && !std::isinf(s0) && !std::isnan(s0) && s0 != 0.0f &&
        !std::signbit(s0))
      match = true;
    if ((mask & 0x100) && std::isnormal(s0) && s0 > 0)
      match = true;
    if ((mask & 0x200) && std::isinf(s0) && s0 > 0)
      match = true;
    if (match)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

VCmpxClassF64Vop3::VCmpxClassF64Vop3(const MachineInst *inst)
    : Vop3("v_cmpx_class_f64", reinterpret_cast<const OpEncoding *>(inst)),
      vdst(64, OperandType::OPR_EXEC, reinterpret_cast<const OpEncoding *>(inst)->vdst),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      src1(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src1) {
  dst_operands_.emplace_back(&vdst);
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&src1);
}

void VCmpxClassF64Vop3::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    if (inst_.abs & (1u << 0))
      s0 = std::fabs(s0);
    if (inst_.neg & (1u << 0))
      s0 = -s0;
    uint32_t mask = src1.read_lane(wf, lane);
    bool match = false;
    if ((mask & 0x001) && std::isnan(s0) &&
        (std::bit_cast<uint64_t>(s0) & 0x0008000000000000ULL) == 0)
      match = true;
    if ((mask & 0x002) && std::isnan(s0) &&
        (std::bit_cast<uint64_t>(s0) & 0x0008000000000000ULL) != 0)
      match = true;
    if ((mask & 0x004) && std::isinf(s0) && s0 < 0)
      match = true;
    if ((mask & 0x008) && std::isnormal(s0) && s0 < 0)
      match = true;
    if ((mask & 0x010) && !std::isnormal(s0) && !std::isinf(s0) && !std::isnan(s0) && s0 != 0.0 &&
        std::signbit(s0))
      match = true;
    if ((mask & 0x020) && s0 == 0.0 && std::signbit(s0))
      match = true;
    if ((mask & 0x040) && s0 == 0.0 && !std::signbit(s0))
      match = true;
    if ((mask & 0x080) && !std::isnormal(s0) && !std::isinf(s0) && !std::isnan(s0) && s0 != 0.0 &&
        !std::signbit(s0))
      match = true;
    if ((mask & 0x100) && std::isnormal(s0) && s0 > 0)
      match = true;
    if ((mask & 0x200) && std::isinf(s0) && s0 > 0)
      match = true;
    if (match)
      result |= (1ULL << lane);
  }
  wf.set_exec(result);
}

} // namespace rdna3
} // namespace rocjitsu

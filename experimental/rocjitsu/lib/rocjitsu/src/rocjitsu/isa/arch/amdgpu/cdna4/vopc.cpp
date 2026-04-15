// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This file was automatically generated. Do not modify.

#include "rocjitsu/isa/arch/amdgpu/cdna4/vopc.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/except.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace rocjitsu {
namespace cdna4 {

VCmpClassF32Vopc::VCmpClassF32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_class_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpClassF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    uint32_t mask = vsrc1.read_lane(wf, lane);
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

VCmpxClassF32Vopc::VCmpxClassF32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_class_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxClassF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    uint32_t mask = vsrc1.read_lane(wf, lane);
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

VCmpClassF64Vopc::VCmpClassF64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_class_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpClassF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    uint32_t mask = vsrc1.read_lane(wf, lane);
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

VCmpxClassF64Vopc::VCmpxClassF64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_class_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxClassF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    uint32_t mask = vsrc1.read_lane(wf, lane);
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

VCmpClassF16Vopc::VCmpClassF16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_class_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpClassF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0_raw = static_cast<uint16_t>(src0.read_lane(wf, lane));
    float s0 = util::f16_to_f32(s0_raw);
    uint32_t mask = vsrc1.read_lane(wf, lane);
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

VCmpxClassF16Vopc::VCmpxClassF16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_class_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxClassF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0_raw = static_cast<uint16_t>(src0.read_lane(wf, lane));
    float s0 = util::f16_to_f32(s0_raw);
    uint32_t mask = vsrc1.read_lane(wf, lane);
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

VCmpFF16Vopc::VCmpFF16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_f_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpFF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLtF16Vopc::VCmpLtF16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_lt_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpLtF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (s0 < s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpEqF16Vopc::VCmpEqF16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_eq_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpEqF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (s0 == s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLeF16Vopc::VCmpLeF16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_le_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpLeF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (s0 <= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGtF16Vopc::VCmpGtF16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_gt_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpGtF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (s0 > s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLgF16Vopc::VCmpLgF16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_lg_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpLgF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (s0 < s1 || s0 > s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGeF16Vopc::VCmpGeF16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_ge_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpGeF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (s0 >= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpOF16Vopc::VCmpOF16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_o_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpOF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (!std::isnan(s0) && !std::isnan(s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpUF16Vopc::VCmpUF16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_u_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpUF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (std::isnan(s0) || std::isnan(s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNgeF16Vopc::VCmpNgeF16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_nge_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpNgeF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (!(s0 >= s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNlgF16Vopc::VCmpNlgF16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_nlg_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpNlgF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (!(s0 < s1 || s0 > s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNgtF16Vopc::VCmpNgtF16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_ngt_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpNgtF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (!(s0 > s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNleF16Vopc::VCmpNleF16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_nle_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpNleF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (!(s0 <= s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNeqF16Vopc::VCmpNeqF16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_neq_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpNeqF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (s0 != s1 || std::isnan(s0) || std::isnan(s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNltF16Vopc::VCmpNltF16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_nlt_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpNltF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (!(s0 < s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpTruF16Vopc::VCmpTruF16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_tru_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpTruF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc |= (1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpxFF16Vopc::VCmpxFF16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_f_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxFF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    (void)lane;
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxLtF16Vopc::VCmpxLtF16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_lt_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxLtF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (s0 < s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxEqF16Vopc::VCmpxEqF16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_eq_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxEqF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (s0 == s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxLeF16Vopc::VCmpxLeF16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_le_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxLeF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (s0 <= s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxGtF16Vopc::VCmpxGtF16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_gt_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxGtF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (s0 > s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxLgF16Vopc::VCmpxLgF16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_lg_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxLgF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (s0 < s1 || s0 > s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxGeF16Vopc::VCmpxGeF16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_ge_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxGeF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (s0 >= s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxOF16Vopc::VCmpxOF16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_o_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxOF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (!std::isnan(s0) && !std::isnan(s1))
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxUF16Vopc::VCmpxUF16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_u_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxUF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (std::isnan(s0) || std::isnan(s1))
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxNgeF16Vopc::VCmpxNgeF16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_nge_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxNgeF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (!(s0 >= s1))
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxNlgF16Vopc::VCmpxNlgF16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_nlg_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxNlgF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (!(s0 < s1 || s0 > s1))
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxNgtF16Vopc::VCmpxNgtF16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_ngt_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxNgtF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (!(s0 > s1))
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxNleF16Vopc::VCmpxNleF16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_nle_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxNleF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (!(s0 <= s1))
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxNeqF16Vopc::VCmpxNeqF16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_neq_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxNeqF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (s0 != s1 || std::isnan(s0) || std::isnan(s1))
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxNltF16Vopc::VCmpxNltF16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_nlt_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxNltF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = util::f16_to_f32(static_cast<uint16_t>(src0.read_lane(wf, lane)));
    float s1 = util::f16_to_f32(static_cast<uint16_t>(vsrc1.read_lane(wf, lane)));
    if (!(s0 < s1))
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxTruF16Vopc::VCmpxTruF16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_tru_f16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxTruF16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpFF32Vopc::VCmpFF32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_f_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpFF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLtF32Vopc::VCmpLtF32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_lt_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpLtF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (s0 < s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpEqF32Vopc::VCmpEqF32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_eq_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpEqF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (s0 == s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLeF32Vopc::VCmpLeF32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_le_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpLeF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (s0 <= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGtF32Vopc::VCmpGtF32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_gt_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpGtF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (s0 > s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLgF32Vopc::VCmpLgF32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_lg_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpLgF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (s0 < s1 || s0 > s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGeF32Vopc::VCmpGeF32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_ge_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpGeF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (s0 >= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpOF32Vopc::VCmpOF32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_o_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpOF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (!std::isnan(s0) && !std::isnan(s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpUF32Vopc::VCmpUF32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_u_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpUF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (std::isnan(s0) || std::isnan(s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNgeF32Vopc::VCmpNgeF32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_nge_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpNgeF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (!(s0 >= s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNlgF32Vopc::VCmpNlgF32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_nlg_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpNlgF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (!(s0 < s1 || s0 > s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNgtF32Vopc::VCmpNgtF32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_ngt_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpNgtF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (!(s0 > s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNleF32Vopc::VCmpNleF32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_nle_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpNleF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (!(s0 <= s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNeqF32Vopc::VCmpNeqF32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_neq_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpNeqF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (s0 != s1 || std::isnan(s0) || std::isnan(s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNltF32Vopc::VCmpNltF32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_nlt_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpNltF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (!(s0 < s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpTruF32Vopc::VCmpTruF32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_tru_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpTruF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc |= (1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpxFF32Vopc::VCmpxFF32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_f_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxFF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    (void)lane;
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxLtF32Vopc::VCmpxLtF32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_lt_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxLtF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (s0 < s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxEqF32Vopc::VCmpxEqF32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_eq_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxEqF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (s0 == s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxLeF32Vopc::VCmpxLeF32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_le_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxLeF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (s0 <= s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxGtF32Vopc::VCmpxGtF32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_gt_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxGtF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (s0 > s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxLgF32Vopc::VCmpxLgF32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_lg_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxLgF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (s0 < s1 || s0 > s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxGeF32Vopc::VCmpxGeF32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_ge_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxGeF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (s0 >= s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxOF32Vopc::VCmpxOF32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_o_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxOF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (!std::isnan(s0) && !std::isnan(s1))
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxUF32Vopc::VCmpxUF32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_u_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxUF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (std::isnan(s0) || std::isnan(s1))
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxNgeF32Vopc::VCmpxNgeF32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_nge_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxNgeF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (!(s0 >= s1))
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxNlgF32Vopc::VCmpxNlgF32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_nlg_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxNlgF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (!(s0 < s1 || s0 > s1))
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxNgtF32Vopc::VCmpxNgtF32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_ngt_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxNgtF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (!(s0 > s1))
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxNleF32Vopc::VCmpxNleF32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_nle_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxNleF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (!(s0 <= s1))
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxNeqF32Vopc::VCmpxNeqF32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_neq_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxNeqF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (s0 != s1 || std::isnan(s0) || std::isnan(s1))
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxNltF32Vopc::VCmpxNltF32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_nlt_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxNltF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float s0 = std::bit_cast<float>(src0.read_lane(wf, lane));
    float s1 = std::bit_cast<float>(vsrc1.read_lane(wf, lane));
    if (!(s0 < s1))
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxTruF32Vopc::VCmpxTruF32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_tru_f32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxTruF32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpFF64Vopc::VCmpFF64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_f_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpFF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLtF64Vopc::VCmpLtF64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_lt_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpLtF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (s0 < s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpEqF64Vopc::VCmpEqF64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_eq_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpEqF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (s0 == s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLeF64Vopc::VCmpLeF64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_le_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpLeF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (s0 <= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGtF64Vopc::VCmpGtF64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_gt_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpGtF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (s0 > s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLgF64Vopc::VCmpLgF64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_lg_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpLgF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (s0 < s1 || s0 > s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGeF64Vopc::VCmpGeF64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_ge_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpGeF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (s0 >= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpOF64Vopc::VCmpOF64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_o_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpOF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (!std::isnan(s0) && !std::isnan(s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpUF64Vopc::VCmpUF64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_u_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpUF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (std::isnan(s0) || std::isnan(s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNgeF64Vopc::VCmpNgeF64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_nge_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpNgeF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (!(s0 >= s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNlgF64Vopc::VCmpNlgF64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_nlg_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpNlgF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (!(s0 < s1 || s0 > s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNgtF64Vopc::VCmpNgtF64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_ngt_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpNgtF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (!(s0 > s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNleF64Vopc::VCmpNleF64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_nle_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpNleF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (!(s0 <= s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNeqF64Vopc::VCmpNeqF64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_neq_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpNeqF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (s0 != s1 || std::isnan(s0) || std::isnan(s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNltF64Vopc::VCmpNltF64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_nlt_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpNltF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (!(s0 < s1))
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpTruF64Vopc::VCmpTruF64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_tru_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpTruF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc |= (1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpxFF64Vopc::VCmpxFF64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_f_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxFF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    (void)lane;
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxLtF64Vopc::VCmpxLtF64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_lt_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxLtF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (s0 < s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxEqF64Vopc::VCmpxEqF64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_eq_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxEqF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (s0 == s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxLeF64Vopc::VCmpxLeF64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_le_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxLeF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (s0 <= s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxGtF64Vopc::VCmpxGtF64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_gt_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxGtF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (s0 > s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxLgF64Vopc::VCmpxLgF64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_lg_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxLgF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (s0 < s1 || s0 > s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxGeF64Vopc::VCmpxGeF64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_ge_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxGeF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (s0 >= s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxOF64Vopc::VCmpxOF64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_o_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxOF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (!std::isnan(s0) && !std::isnan(s1))
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxUF64Vopc::VCmpxUF64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_u_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxUF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (std::isnan(s0) || std::isnan(s1))
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxNgeF64Vopc::VCmpxNgeF64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_nge_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxNgeF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (!(s0 >= s1))
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxNlgF64Vopc::VCmpxNlgF64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_nlg_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxNlgF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (!(s0 < s1 || s0 > s1))
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxNgtF64Vopc::VCmpxNgtF64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_ngt_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxNgtF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (!(s0 > s1))
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxNleF64Vopc::VCmpxNleF64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_nle_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxNleF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (!(s0 <= s1))
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxNeqF64Vopc::VCmpxNeqF64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_neq_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxNeqF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (s0 != s1 || std::isnan(s0) || std::isnan(s1))
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxNltF64Vopc::VCmpxNltF64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_nlt_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxNltF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    double s0 = std::bit_cast<double>(src0.read_lane64(wf, lane));
    double s1 = std::bit_cast<double>(vsrc1.read_lane64(wf, lane));
    if (!(s0 < s1))
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxTruF64Vopc::VCmpxTruF64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_tru_f64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxTruF64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpFI16Vopc::VCmpFI16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_f_i16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpFI16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLtI16Vopc::VCmpLtI16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_lt_i16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpLtI16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t s0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t s1 = static_cast<int16_t>(vsrc1.read_lane(wf, lane) & 0xFFFF);
    if (s0 < s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpEqI16Vopc::VCmpEqI16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_eq_i16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpEqI16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t s0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t s1 = static_cast<int16_t>(vsrc1.read_lane(wf, lane) & 0xFFFF);
    if (s0 == s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLeI16Vopc::VCmpLeI16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_le_i16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpLeI16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t s0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t s1 = static_cast<int16_t>(vsrc1.read_lane(wf, lane) & 0xFFFF);
    if (s0 <= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGtI16Vopc::VCmpGtI16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_gt_i16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpGtI16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t s0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t s1 = static_cast<int16_t>(vsrc1.read_lane(wf, lane) & 0xFFFF);
    if (s0 > s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNeI16Vopc::VCmpNeI16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_ne_i16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpNeI16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t s0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t s1 = static_cast<int16_t>(vsrc1.read_lane(wf, lane) & 0xFFFF);
    if (s0 != s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGeI16Vopc::VCmpGeI16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_ge_i16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpGeI16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t s0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t s1 = static_cast<int16_t>(vsrc1.read_lane(wf, lane) & 0xFFFF);
    if (s0 >= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpTI16Vopc::VCmpTI16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_t_i16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpTI16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc |= (1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpFU16Vopc::VCmpFU16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_f_u16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpFU16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLtU16Vopc::VCmpLtU16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_lt_u16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpLtU16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t s1 = static_cast<uint16_t>(vsrc1.read_lane(wf, lane));
    if (s0 < s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpEqU16Vopc::VCmpEqU16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_eq_u16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpEqU16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t s1 = static_cast<uint16_t>(vsrc1.read_lane(wf, lane));
    if (s0 == s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLeU16Vopc::VCmpLeU16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_le_u16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpLeU16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t s1 = static_cast<uint16_t>(vsrc1.read_lane(wf, lane));
    if (s0 <= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGtU16Vopc::VCmpGtU16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_gt_u16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpGtU16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t s1 = static_cast<uint16_t>(vsrc1.read_lane(wf, lane));
    if (s0 > s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNeU16Vopc::VCmpNeU16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_ne_u16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpNeU16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t s1 = static_cast<uint16_t>(vsrc1.read_lane(wf, lane));
    if (s0 != s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGeU16Vopc::VCmpGeU16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_ge_u16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpGeU16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t s1 = static_cast<uint16_t>(vsrc1.read_lane(wf, lane));
    if (s0 >= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpTU16Vopc::VCmpTU16Vopc(const MachineInst *inst)
    : Vopc("v_cmp_t_u16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpTU16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc |= (1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpxFI16Vopc::VCmpxFI16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_f_i16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxFI16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    (void)lane;
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxLtI16Vopc::VCmpxLtI16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_lt_i16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxLtI16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t s0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t s1 = static_cast<int16_t>(vsrc1.read_lane(wf, lane) & 0xFFFF);
    if (s0 < s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxEqI16Vopc::VCmpxEqI16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_eq_i16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxEqI16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t s0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t s1 = static_cast<int16_t>(vsrc1.read_lane(wf, lane) & 0xFFFF);
    if (s0 == s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxLeI16Vopc::VCmpxLeI16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_le_i16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxLeI16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t s0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t s1 = static_cast<int16_t>(vsrc1.read_lane(wf, lane) & 0xFFFF);
    if (s0 <= s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxGtI16Vopc::VCmpxGtI16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_gt_i16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxGtI16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t s0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t s1 = static_cast<int16_t>(vsrc1.read_lane(wf, lane) & 0xFFFF);
    if (s0 > s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxNeI16Vopc::VCmpxNeI16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_ne_i16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxNeI16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t s0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t s1 = static_cast<int16_t>(vsrc1.read_lane(wf, lane) & 0xFFFF);
    if (s0 != s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxGeI16Vopc::VCmpxGeI16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_ge_i16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxGeI16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int16_t s0 = static_cast<int16_t>(src0.read_lane(wf, lane) & 0xFFFF);
    int16_t s1 = static_cast<int16_t>(vsrc1.read_lane(wf, lane) & 0xFFFF);
    if (s0 >= s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxTI16Vopc::VCmpxTI16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_t_i16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxTI16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxFU16Vopc::VCmpxFU16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_f_u16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxFU16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    (void)lane;
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxLtU16Vopc::VCmpxLtU16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_lt_u16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxLtU16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t s1 = static_cast<uint16_t>(vsrc1.read_lane(wf, lane));
    if (s0 < s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxEqU16Vopc::VCmpxEqU16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_eq_u16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxEqU16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t s1 = static_cast<uint16_t>(vsrc1.read_lane(wf, lane));
    if (s0 == s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxLeU16Vopc::VCmpxLeU16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_le_u16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxLeU16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t s1 = static_cast<uint16_t>(vsrc1.read_lane(wf, lane));
    if (s0 <= s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxGtU16Vopc::VCmpxGtU16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_gt_u16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxGtU16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t s1 = static_cast<uint16_t>(vsrc1.read_lane(wf, lane));
    if (s0 > s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxNeU16Vopc::VCmpxNeU16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_ne_u16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxNeU16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t s1 = static_cast<uint16_t>(vsrc1.read_lane(wf, lane));
    if (s0 != s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxGeU16Vopc::VCmpxGeU16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_ge_u16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxGeU16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint16_t s0 = static_cast<uint16_t>(src0.read_lane(wf, lane));
    uint16_t s1 = static_cast<uint16_t>(vsrc1.read_lane(wf, lane));
    if (s0 >= s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxTU16Vopc::VCmpxTU16Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_t_u16", reinterpret_cast<const OpEncoding *>(inst)),
      src0(16, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(16, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        16, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxTU16Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpFI32Vopc::VCmpFI32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_f_i32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpFI32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLtI32Vopc::VCmpLtI32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_lt_i32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpLtI32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t s1 = static_cast<int32_t>(vsrc1.read_lane(wf, lane));
    if (s0 < s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpEqI32Vopc::VCmpEqI32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_eq_i32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpEqI32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t s1 = static_cast<int32_t>(vsrc1.read_lane(wf, lane));
    if (s0 == s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLeI32Vopc::VCmpLeI32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_le_i32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpLeI32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t s1 = static_cast<int32_t>(vsrc1.read_lane(wf, lane));
    if (s0 <= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGtI32Vopc::VCmpGtI32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_gt_i32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpGtI32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t s1 = static_cast<int32_t>(vsrc1.read_lane(wf, lane));
    if (s0 > s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNeI32Vopc::VCmpNeI32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_ne_i32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpNeI32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t s1 = static_cast<int32_t>(vsrc1.read_lane(wf, lane));
    if (s0 != s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGeI32Vopc::VCmpGeI32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_ge_i32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpGeI32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t s1 = static_cast<int32_t>(vsrc1.read_lane(wf, lane));
    if (s0 >= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpTI32Vopc::VCmpTI32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_t_i32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpTI32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc |= (1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpFU32Vopc::VCmpFU32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_f_u32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpFU32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLtU32Vopc::VCmpLtU32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_lt_u32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpLtU32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane);
    uint32_t s1 = vsrc1.read_lane(wf, lane);
    if (s0 < s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpEqU32Vopc::VCmpEqU32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_eq_u32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpEqU32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane);
    uint32_t s1 = vsrc1.read_lane(wf, lane);
    if (s0 == s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLeU32Vopc::VCmpLeU32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_le_u32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpLeU32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane);
    uint32_t s1 = vsrc1.read_lane(wf, lane);
    if (s0 <= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGtU32Vopc::VCmpGtU32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_gt_u32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpGtU32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane);
    uint32_t s1 = vsrc1.read_lane(wf, lane);
    if (s0 > s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNeU32Vopc::VCmpNeU32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_ne_u32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpNeU32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane);
    uint32_t s1 = vsrc1.read_lane(wf, lane);
    if (s0 != s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGeU32Vopc::VCmpGeU32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_ge_u32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpGeU32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane);
    uint32_t s1 = vsrc1.read_lane(wf, lane);
    if (s0 >= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpTU32Vopc::VCmpTU32Vopc(const MachineInst *inst)
    : Vopc("v_cmp_t_u32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpTU32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc |= (1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpxFI32Vopc::VCmpxFI32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_f_i32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxFI32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    (void)lane;
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxLtI32Vopc::VCmpxLtI32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_lt_i32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxLtI32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t s1 = static_cast<int32_t>(vsrc1.read_lane(wf, lane));
    if (s0 < s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxEqI32Vopc::VCmpxEqI32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_eq_i32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxEqI32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t s1 = static_cast<int32_t>(vsrc1.read_lane(wf, lane));
    if (s0 == s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxLeI32Vopc::VCmpxLeI32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_le_i32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxLeI32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t s1 = static_cast<int32_t>(vsrc1.read_lane(wf, lane));
    if (s0 <= s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxGtI32Vopc::VCmpxGtI32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_gt_i32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxGtI32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t s1 = static_cast<int32_t>(vsrc1.read_lane(wf, lane));
    if (s0 > s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxNeI32Vopc::VCmpxNeI32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_ne_i32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxNeI32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t s1 = static_cast<int32_t>(vsrc1.read_lane(wf, lane));
    if (s0 != s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxGeI32Vopc::VCmpxGeI32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_ge_i32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxGeI32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int32_t s0 = static_cast<int32_t>(src0.read_lane(wf, lane));
    int32_t s1 = static_cast<int32_t>(vsrc1.read_lane(wf, lane));
    if (s0 >= s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxTI32Vopc::VCmpxTI32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_t_i32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxTI32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxFU32Vopc::VCmpxFU32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_f_u32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxFU32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    (void)lane;
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxLtU32Vopc::VCmpxLtU32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_lt_u32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxLtU32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane);
    uint32_t s1 = vsrc1.read_lane(wf, lane);
    if (s0 < s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxEqU32Vopc::VCmpxEqU32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_eq_u32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxEqU32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane);
    uint32_t s1 = vsrc1.read_lane(wf, lane);
    if (s0 == s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxLeU32Vopc::VCmpxLeU32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_le_u32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxLeU32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane);
    uint32_t s1 = vsrc1.read_lane(wf, lane);
    if (s0 <= s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxGtU32Vopc::VCmpxGtU32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_gt_u32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxGtU32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane);
    uint32_t s1 = vsrc1.read_lane(wf, lane);
    if (s0 > s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxNeU32Vopc::VCmpxNeU32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_ne_u32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxNeU32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane);
    uint32_t s1 = vsrc1.read_lane(wf, lane);
    if (s0 != s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxGeU32Vopc::VCmpxGeU32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_ge_u32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxGeU32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint32_t s0 = src0.read_lane(wf, lane);
    uint32_t s1 = vsrc1.read_lane(wf, lane);
    if (s0 >= s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxTU32Vopc::VCmpxTU32Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_t_u32", reinterpret_cast<const OpEncoding *>(inst)),
      src0(32, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(32, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        32, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxTU32Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpFI64Vopc::VCmpFI64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_f_i64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpFI64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLtI64Vopc::VCmpLtI64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_lt_i64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpLtI64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int64_t s0 = static_cast<int64_t>(src0.read_lane64(wf, lane));
    int64_t s1 = static_cast<int64_t>(vsrc1.read_lane64(wf, lane));
    if (s0 < s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpEqI64Vopc::VCmpEqI64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_eq_i64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpEqI64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int64_t s0 = static_cast<int64_t>(src0.read_lane64(wf, lane));
    int64_t s1 = static_cast<int64_t>(vsrc1.read_lane64(wf, lane));
    if (s0 == s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLeI64Vopc::VCmpLeI64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_le_i64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpLeI64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int64_t s0 = static_cast<int64_t>(src0.read_lane64(wf, lane));
    int64_t s1 = static_cast<int64_t>(vsrc1.read_lane64(wf, lane));
    if (s0 <= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGtI64Vopc::VCmpGtI64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_gt_i64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpGtI64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int64_t s0 = static_cast<int64_t>(src0.read_lane64(wf, lane));
    int64_t s1 = static_cast<int64_t>(vsrc1.read_lane64(wf, lane));
    if (s0 > s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNeI64Vopc::VCmpNeI64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_ne_i64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpNeI64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int64_t s0 = static_cast<int64_t>(src0.read_lane64(wf, lane));
    int64_t s1 = static_cast<int64_t>(vsrc1.read_lane64(wf, lane));
    if (s0 != s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGeI64Vopc::VCmpGeI64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_ge_i64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpGeI64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int64_t s0 = static_cast<int64_t>(src0.read_lane64(wf, lane));
    int64_t s1 = static_cast<int64_t>(vsrc1.read_lane64(wf, lane));
    if (s0 >= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpTI64Vopc::VCmpTI64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_t_i64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpTI64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc |= (1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpFU64Vopc::VCmpFU64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_f_u64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpFU64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLtU64Vopc::VCmpLtU64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_lt_u64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpLtU64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t s0 = src0.read_lane64(wf, lane);
    uint64_t s1 = vsrc1.read_lane64(wf, lane);
    if (s0 < s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpEqU64Vopc::VCmpEqU64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_eq_u64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpEqU64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t s0 = src0.read_lane64(wf, lane);
    uint64_t s1 = vsrc1.read_lane64(wf, lane);
    if (s0 == s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpLeU64Vopc::VCmpLeU64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_le_u64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpLeU64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t s0 = src0.read_lane64(wf, lane);
    uint64_t s1 = vsrc1.read_lane64(wf, lane);
    if (s0 <= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGtU64Vopc::VCmpGtU64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_gt_u64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpGtU64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t s0 = src0.read_lane64(wf, lane);
    uint64_t s1 = vsrc1.read_lane64(wf, lane);
    if (s0 > s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpNeU64Vopc::VCmpNeU64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_ne_u64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpNeU64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t s0 = src0.read_lane64(wf, lane);
    uint64_t s1 = vsrc1.read_lane64(wf, lane);
    if (s0 != s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpGeU64Vopc::VCmpGeU64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_ge_u64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpGeU64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t s0 = src0.read_lane64(wf, lane);
    uint64_t s1 = vsrc1.read_lane64(wf, lane);
    if (s0 >= s1)
      vcc |= (1ULL << lane);
    else
      vcc &= ~(1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpTU64Vopc::VCmpTU64Vopc(const MachineInst *inst)
    : Vopc("v_cmp_t_u64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpTU64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    vcc |= (1ULL << lane);
  }
  wf.set_vcc(vcc);
}

VCmpxFI64Vopc::VCmpxFI64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_f_i64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxFI64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    (void)lane;
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxLtI64Vopc::VCmpxLtI64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_lt_i64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxLtI64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int64_t s0 = static_cast<int64_t>(src0.read_lane64(wf, lane));
    int64_t s1 = static_cast<int64_t>(vsrc1.read_lane64(wf, lane));
    if (s0 < s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxEqI64Vopc::VCmpxEqI64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_eq_i64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxEqI64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int64_t s0 = static_cast<int64_t>(src0.read_lane64(wf, lane));
    int64_t s1 = static_cast<int64_t>(vsrc1.read_lane64(wf, lane));
    if (s0 == s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxLeI64Vopc::VCmpxLeI64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_le_i64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxLeI64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int64_t s0 = static_cast<int64_t>(src0.read_lane64(wf, lane));
    int64_t s1 = static_cast<int64_t>(vsrc1.read_lane64(wf, lane));
    if (s0 <= s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxGtI64Vopc::VCmpxGtI64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_gt_i64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxGtI64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int64_t s0 = static_cast<int64_t>(src0.read_lane64(wf, lane));
    int64_t s1 = static_cast<int64_t>(vsrc1.read_lane64(wf, lane));
    if (s0 > s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxNeI64Vopc::VCmpxNeI64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_ne_i64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxNeI64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int64_t s0 = static_cast<int64_t>(src0.read_lane64(wf, lane));
    int64_t s1 = static_cast<int64_t>(vsrc1.read_lane64(wf, lane));
    if (s0 != s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxGeI64Vopc::VCmpxGeI64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_ge_i64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxGeI64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    int64_t s0 = static_cast<int64_t>(src0.read_lane64(wf, lane));
    int64_t s1 = static_cast<int64_t>(vsrc1.read_lane64(wf, lane));
    if (s0 >= s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxTI64Vopc::VCmpxTI64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_t_i64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxTI64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxFU64Vopc::VCmpxFU64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_f_u64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxFU64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    (void)lane;
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxLtU64Vopc::VCmpxLtU64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_lt_u64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxLtU64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t s0 = src0.read_lane64(wf, lane);
    uint64_t s1 = vsrc1.read_lane64(wf, lane);
    if (s0 < s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxEqU64Vopc::VCmpxEqU64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_eq_u64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxEqU64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t s0 = src0.read_lane64(wf, lane);
    uint64_t s1 = vsrc1.read_lane64(wf, lane);
    if (s0 == s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxLeU64Vopc::VCmpxLeU64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_le_u64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxLeU64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t s0 = src0.read_lane64(wf, lane);
    uint64_t s1 = vsrc1.read_lane64(wf, lane);
    if (s0 <= s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxGtU64Vopc::VCmpxGtU64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_gt_u64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxGtU64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t s0 = src0.read_lane64(wf, lane);
    uint64_t s1 = vsrc1.read_lane64(wf, lane);
    if (s0 > s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxNeU64Vopc::VCmpxNeU64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_ne_u64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxNeU64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t s0 = src0.read_lane64(wf, lane);
    uint64_t s1 = vsrc1.read_lane64(wf, lane);
    if (s0 != s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxGeU64Vopc::VCmpxGeU64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_ge_u64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxGeU64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    uint64_t s0 = src0.read_lane64(wf, lane);
    uint64_t s1 = vsrc1.read_lane64(wf, lane);
    if (s0 >= s1)
      result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

VCmpxTU64Vopc::VCmpxTU64Vopc(const MachineInst *inst)
    : Vopc("v_cmpx_t_u64", reinterpret_cast<const OpEncoding *>(inst)),
      src0(64, OperandType::OPR_SRC, reinterpret_cast<const OpEncoding *>(inst)->src0),
      vsrc1(64, OperandType::OPR_VGPR, reinterpret_cast<const OpEncoding *>(inst)->vsrc1) {
  src_operands_.emplace_back(&src0);
  src_operands_.emplace_back(&vsrc1);
  if (reinterpret_cast<const OpEncoding *>(inst)->src0 == 255)
    src0 = Operand(
        64, OperandType::OPR_SIMM32,
        static_cast<int>(reinterpret_cast<const VopcInstLiteralMachineInst *>(inst)->simm32));
}

void VCmpxTU64Vopc::execute(amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  uint64_t result = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    result |= (1ULL << lane);
  }
  wf.set_vcc(result);
  wf.set_exec(result);
}

} // namespace cdna4
} // namespace rocjitsu

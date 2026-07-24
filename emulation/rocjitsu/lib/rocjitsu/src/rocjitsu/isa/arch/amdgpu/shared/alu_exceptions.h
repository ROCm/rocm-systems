// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_ALU_EXCEPTIONS_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_ALU_EXCEPTIONS_H_

#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include <bit>
#include <cmath>
#include <cstdint>

namespace rocjitsu::amdgpu {

inline constexpr uint32_t kAluExceptionModeMask = 0x7fu << 12;
inline constexpr uint32_t kAluExceptionTrapstsMask = 0x7fu;

inline uint32_t classify_mul_f32(float lhs, float rhs) {
  uint32_t causes = 0;
  if (std::fpclassify(lhs) == FP_SUBNORMAL || std::fpclassify(rhs) == FP_SUBNORMAL)
    causes |= 1u << 1;
  const long double exact = static_cast<long double>(lhs) * static_cast<long double>(rhs);
  const float result = lhs * rhs;
  if (std::isfinite(lhs) && std::isfinite(rhs) && std::isinf(result))
    causes |= 1u << 3;
  if (exact != 0.0L && (result == 0.0f || std::fpclassify(result) == FP_SUBNORMAL))
    causes |= 1u << 4;
  if (std::isfinite(exact) && static_cast<long double>(result) != exact)
    causes |= 1u << 5;
  return causes;
}

template <typename Inst> uint32_t classify_mul_f32_vop2(const Inst &inst, Wavefront &wf) {
  uint32_t causes = 0;
  const uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    const float lhs = std::bit_cast<float>(RegisterAccess(wf).read_lane(inst.src0, lane));
    const float rhs = std::bit_cast<float>(RegisterAccess(wf).read_lane(inst.vsrc1, lane));
    causes |= classify_mul_f32(lhs, rhs);
  }
  return causes;
}

template <typename Inst> uint32_t classify_mul_f32_vop3(const Inst &inst, Wavefront &wf) {
  uint32_t causes = 0;
  const uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float lhs = std::bit_cast<float>(RegisterAccess(wf).read_lane(inst.src0, lane));
    float rhs = std::bit_cast<float>(RegisterAccess(wf).read_lane(inst.src1, lane));
    if (inst.inst_.abs & 1u)
      lhs = std::fabs(lhs);
    if (inst.inst_.neg & 1u)
      lhs = -lhs;
    if (inst.inst_.abs & 2u)
      rhs = std::fabs(rhs);
    if (inst.inst_.neg & 2u)
      rhs = -rhs;
    float result = lhs * rhs;
    if (inst.inst_.omod == 1)
      result *= 2.0f;
    else if (inst.inst_.omod == 2)
      result *= 4.0f;
    else if (inst.inst_.omod == 3)
      result *= 0.5f;
    causes |= classify_mul_f32(lhs, rhs);
    if (result != lhs * rhs)
      causes |= 1u << 5;
  }
  return causes;
}

template <typename Inst> uint32_t classify_sqrt_f32_vop1(const Inst &inst, Wavefront &wf) {
  const uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane)
    if ((exec & (1ULL << lane)) &&
        std::bit_cast<float>(RegisterAccess(wf).read_lane(inst.src0, lane)) < 0.0f)
      return 1u;
  return 0;
}

template <typename Inst> uint32_t classify_sqrt_f32_vop3(const Inst &inst, Wavefront &wf) {
  const uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float source = std::bit_cast<float>(RegisterAccess(wf).read_lane(inst.src0, lane));
    if (inst.inst_.abs & 1u)
      source = std::fabs(source);
    if (inst.inst_.neg & 1u)
      source = -source;
    if (source < 0.0f)
      return 1u;
  }
  return 0;
}

template <typename Inst>
uint32_t classify_div_fixup_f32_exceptions(const Inst &inst, Wavefront &wf) {
  const uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    float denominator = std::bit_cast<float>(RegisterAccess(wf).read_lane(inst.src1, lane));
    float numerator = std::bit_cast<float>(RegisterAccess(wf).read_lane(inst.src2, lane));
    if (inst.inst_.abs & 2u)
      denominator = std::fabs(denominator);
    if (inst.inst_.neg & 2u)
      denominator = -denominator;
    if (inst.inst_.abs & 4u)
      numerator = std::fabs(numerator);
    if (inst.inst_.neg & 4u)
      numerator = -numerator;
    if (denominator == 0.0f && numerator != 0.0f)
      return 1u << 2;
  }
  return 0;
}

template <typename Inst>
uint32_t classify_rcp_iflag_f32_exceptions(const Inst &inst, Wavefront &wf) {
  const uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane)
    if ((exec & (1ULL << lane)) &&
        std::bit_cast<float>(RegisterAccess(wf).read_lane(inst.src0, lane)) == 0.0f)
      return 1u << 6;
  return 0;
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_AMDGPU_SHARED_ALU_EXCEPTIONS_H_

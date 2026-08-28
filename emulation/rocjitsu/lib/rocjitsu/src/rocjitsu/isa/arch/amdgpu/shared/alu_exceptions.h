// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_ALU_EXCEPTIONS_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_ALU_EXCEPTIONS_H_

#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode_no_fenv.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include <bit>
#include <cmath>
#include <cstdint>

namespace rocjitsu::amdgpu {

inline constexpr uint32_t kAluExceptionModeShift = 12;
inline constexpr uint32_t kAluExceptionModeMask = 0x7fu << kAluExceptionModeShift;
inline constexpr uint32_t kAluExceptionTrapstsMask = 0x7fu;

/// @brief Return the enabled ALU trap causes in EXCP_FLAG bit positions.
///
/// GFX12 moved these enables out of MODE[18:12], where those bits now select
/// VGPR high banks, and into TRAP_CTRL[6:0]. Keeping the normalized mask here
/// prevents exception checks from treating a debugger's trap enables as VGPR
/// selectors, or a shader's VGPR selectors as enabled exceptions.
inline uint32_t alu_exception_trap_enables(const Wavefront &wf) {
  return wf.uses_separate_trap_ctrl()
             ? wf.gfx12_trap_ctrl_raw() & kAluExceptionTrapstsMask
             : (wf.mode_raw() & kAluExceptionModeMask) >> kAluExceptionModeShift;
}

// Every classifier below reports what it found through wf.raise_alu_causes()
// as well as returning it. The generated call sites OR the return value into
// TRAPSTS, which is architecturally sticky and so cannot tell the CU whether
// the current instruction raised a cause or merely inherited a bit some
// earlier instruction latched. Trap delivery needs the former, so a classifier
// that grows a new early return has to report on that path too.

/// @brief EXCP causes raised by `lhs * rhs`, optionally scaled by an output
/// modifier.
/// @details @p omod_scale must be applied to the exact and the rounded value
/// alike. Deriving INEXACT by comparing the scaled result against the unscaled
/// product instead reports every nonzero product as inexact the moment an
/// output modifier is present, because scaling by 2 changes the value it is
/// being compared against.
inline uint32_t classify_mul_f32(float lhs, float rhs, float omod_scale = 1.0f) {
  uint32_t causes = 0;
  if (std::fpclassify(lhs) == FP_SUBNORMAL || std::fpclassify(rhs) == FP_SUBNORMAL)
    causes |= 1u << 1;
  const long double exact =
      static_cast<long double>(lhs) * static_cast<long double>(rhs) * omod_scale;
  const float result = (lhs * rhs) * omod_scale;
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
  wf.raise_alu_causes(causes);
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
    // OMOD scales the result, so it has to take part in deriving the causes
    // rather than being compared against the unscaled product afterwards.
    const uint32_t omod = fp_mode::effective_omod(wf.cu().arch(), wf.fp_denorm_mode_f32(),
                                                  wf.ieee_mode(), inst.inst_.omod);
    float omod_scale = 1.0f;
    if (omod == 1)
      omod_scale = 2.0f;
    else if (omod == 2)
      omod_scale = 4.0f;
    else if (omod == 3)
      omod_scale = 0.5f;
    uint32_t lane_causes = classify_mul_f32(lhs, rhs, omod_scale);
    if (omod != 0 &&
        (wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA4 || wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA5))
      lane_causes &= ~((1u << 4) | (1u << 5));
    causes |= lane_causes;
  }
  wf.raise_alu_causes(causes);
  return causes;
}

template <typename Inst> uint32_t classify_sqrt_f32_vop1(const Inst &inst, Wavefront &wf) {
  const uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane)
    if ((exec & (1ULL << lane)) &&
        std::bit_cast<float>(RegisterAccess(wf).read_lane(inst.src0, lane)) < 0.0f) {
      wf.raise_alu_causes(1u);
      return 1u;
    }
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
    if (source < 0.0f) {
      wf.raise_alu_causes(1u);
      return 1u;
    }
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
    if (denominator == 0.0f && numerator != 0.0f) {
      wf.raise_alu_causes(1u << 2);
      return 1u << 2;
    }
  }
  return 0;
}

template <typename Inst>
uint32_t classify_rcp_iflag_f32_exceptions(const Inst &inst, Wavefront &wf) {
  const uint64_t exec = wf.exec();
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane)
    if ((exec & (1ULL << lane)) &&
        std::bit_cast<float>(RegisterAccess(wf).read_lane(inst.src0, lane)) == 0.0f) {
      wf.raise_alu_causes(1u << 6);
      return 1u << 6;
    }
  return 0;
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_AMDGPU_SHARED_ALU_EXCEPTIONS_H_

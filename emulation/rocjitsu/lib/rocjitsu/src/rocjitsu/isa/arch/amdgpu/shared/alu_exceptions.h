// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_ALU_EXCEPTIONS_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_ALU_EXCEPTIONS_H_

#include "rocjitsu/isa/arch/amdgpu/cdna1/alu_status.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/valu.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <type_traits>

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

namespace detail {

// A host F32-to-F64 conversion can flush subnormals under DAZ. Form those
// values from their integer significands; nonzero finite F32 values are normal in F64.
inline double widen_alu_f32(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  const uint32_t magnitude = bits & 0x7fffffffu;
  if (magnitude < 0x00800000u) {
    const double widened = static_cast<double>(magnitude) * 0x1p-149;
    return (bits >> 31) ? -widened : widened;
  }
  return static_cast<double>(value);
}

template <typename Lhs, typename Rhs, typename Classify>
uint32_t classify_mul_lanes(const Lhs &lhs, const Rhs &rhs, Wavefront &wf, Classify classify) {
  uint32_t causes = 0;
  const uint64_t exec = wf.exec();
  RegisterAccess regs(wf);
  auto visit = [&](auto read_lhs, auto read_rhs) {
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
      if (exec & (1ULL << lane)) {
        causes |=
            classify(std::bit_cast<float>(read_lhs(lane)), std::bit_cast<float>(read_rhs(lane)));
      }
    }
  };
  // Views resolve contiguous VGPR storage and scalar broadcasts once while
  // retaining register ownership checks and plugin-visible active-lane reads.
  // Delegates and sub-dword selections keep their per-lane operand semantics.
  if (exec && lhs.simd_capable() && rhs.simd_capable()) {
    auto lhs_view = regs.read_operand(lhs, exec);
    auto rhs_view = regs.read_operand(rhs, exec);
    visit([&](uint32_t lane) { return lhs_view.lane(lane); },
          [&](uint32_t lane) { return rhs_view.lane(lane); });
  } else {
    visit([&](uint32_t lane) { return regs.read_lane(lhs, lane); },
          [&](uint32_t lane) { return regs.read_lane(rhs, lane); });
  }
  wf.raise_alu_causes(causes);
  return causes;
}

} // namespace detail

/// @brief EXCP causes raised by `lhs * rhs`, optionally scaled by an output
/// modifier.
/// @details @p omod_scale must be applied to the exact and the rounded value
/// alike. Deriving INEXACT by comparing the scaled result against the unscaled
/// product instead reports every nonzero product as inexact the moment an
/// output modifier is present, because scaling by 2 changes the value it is
/// being compared against.
/// @param omod_scale One of the architectural scales: 1, 2, 4, or 0.5.
inline uint32_t classify_mul_f32(float lhs, float rhs, float omod_scale = 1.0f) {
  assert(omod_scale == 1.0f || omod_scale == 2.0f || omod_scale == 4.0f || omod_scale == 0.5f);
  uint32_t causes = 0;
  if (std::fpclassify(lhs) == FP_SUBNORMAL || std::fpclassify(rhs) == FP_SUBNORMAL) {
    causes |= 1u << 1;
  }
  // An F32 product needs at most 48 significand bits. The power-of-two OMOD
  // scales (1, 2, 4, 0.5) preserve that precision and stay normal in F64.
  const double exact = detail::widen_alu_f32(lhs) * detail::widen_alu_f32(rhs) * omod_scale;
  const float result = (lhs * rhs) * omod_scale;
  if (std::isfinite(lhs) && std::isfinite(rhs) && std::isinf(result)) {
    causes |= 1u << 3;
  }
  if (exact != 0.0 && (result == 0.0f || std::fpclassify(result) == FP_SUBNORMAL)) {
    causes |= 1u << 4;
  }
  if (std::isfinite(exact) && detail::widen_alu_f32(result) != exact) {
    causes |= 1u << 5;
  }
  return causes;
}

template <typename Inst> uint32_t classify_mul_f32_vop2(const Inst &inst, Wavefront &wf) {
  if (wf.cu().arch() == rocjitsu::IsaFamily::Rdna3_5) {
    const uint32_t causes =
        rdna3_5::arithmetic_flags<rdna3_5::FmaOperation::Multiply>(inst, wf, inst.vsrc1);
    wf.raise_alu_causes(causes);
    return causes;
  }
  if (wf.cu().arch() == rocjitsu::IsaFamily::Cdna1) {
    const uint32_t causes = rocjitsu::cdna1::classify_binary_f32<true>(inst, wf, inst.vsrc1);
    wf.raise_alu_causes(causes);
    return causes;
  }
  return detail::classify_mul_lanes(
      inst.src0, inst.vsrc1, wf, [](float lhs, float rhs) { return classify_mul_f32(lhs, rhs); });
}

template <typename Inst> uint32_t classify_mul_f32_vop3(const Inst &inst, Wavefront &wf) {
  if (wf.cu().arch() == rocjitsu::IsaFamily::Rdna3_5) {
    const uint32_t causes =
        rdna3_5::arithmetic_flags<rdna3_5::FmaOperation::Multiply>(inst, wf, inst.src1);
    wf.raise_alu_causes(causes);
    return causes;
  }
  if (wf.cu().arch() == rocjitsu::IsaFamily::Cdna1) {
    const uint32_t causes = rocjitsu::cdna1::classify_binary_f32<true>(inst, wf, inst.src1);
    wf.raise_alu_causes(causes);
    return causes;
  }
  // OMOD scales the exact and rounded products alike. Its policy is uniform
  // across the wave, including which causes the target suppresses.
  const uint32_t omod = fp_mode::effective_omod(wf.cu().arch(), wf.fp_denorm_mode_f32(),
                                                wf.ieee_mode(), inst.inst_.omod);
  const float omod_scale = omod == 1 ? 2.0f : omod == 2 ? 4.0f : omod == 3 ? 0.5f : 1.0f;
  const uint32_t suppressed = omod != 0 && (wf.cu().arch() == rocjitsu::IsaFamily::Rdna4 ||
                                            wf.cu().arch() == rocjitsu::IsaFamily::Cdna5)
                                  ? (1u << 4) | (1u << 5)
                                  : 0;
  return detail::classify_mul_lanes(inst.src0, inst.src1, wf, [&](float lhs, float rhs) {
    if (inst.inst_.abs & 1u) {
      lhs = std::fabs(lhs);
    }
    if (inst.inst_.neg & 1u) {
      lhs = -lhs;
    }
    if (inst.inst_.abs & 2u) {
      rhs = std::fabs(rhs);
    }
    if (inst.inst_.neg & 2u) {
      rhs = -rhs;
    }
    return classify_mul_f32(lhs, rhs, omod_scale) & ~suppressed;
  });
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_AMDGPU_SHARED_ALU_EXCEPTIONS_H_

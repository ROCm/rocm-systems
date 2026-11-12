// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_ALU_EXCEPTIONS_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_ALU_EXCEPTIONS_H_

#include "rocjitsu/isa/arch/amdgpu/cdna1/alu_status.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/valu.h"
#include "rocjitsu/isa/arch/amdgpu/shared/division.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
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
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane)
      if (exec & (1ULL << lane))
        causes |=
            classify(std::bit_cast<float>(read_lhs(lane)), std::bit_cast<float>(read_rhs(lane)));
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
  if (std::fpclassify(lhs) == FP_SUBNORMAL || std::fpclassify(rhs) == FP_SUBNORMAL)
    causes |= 1u << 1;
  // An F32 product needs at most 48 significand bits. The power-of-two OMOD
  // scales (1, 2, 4, 0.5) preserve that precision and stay normal in F64.
  const double exact = detail::widen_alu_f32(lhs) * detail::widen_alu_f32(rhs) * omod_scale;
  const float result = (lhs * rhs) * omod_scale;
  if (std::isfinite(lhs) && std::isfinite(rhs) && std::isinf(result))
    causes |= 1u << 3;
  if (exact != 0.0 && (result == 0.0f || std::fpclassify(result) == FP_SUBNORMAL))
    causes |= 1u << 4;
  if (std::isfinite(exact) && detail::widen_alu_f32(result) != exact)
    causes |= 1u << 5;
  return causes;
}

template <typename Inst> uint32_t classify_mul_f32_vop2(const Inst &inst, Wavefront &wf) {
  if (wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA3_5) {
    const uint32_t causes =
        rdna3_5::arithmetic_flags<rdna3_5::FmaOperation::Multiply>(inst, wf, inst.vsrc1);
    wf.raise_alu_causes(causes);
    return causes;
  }
  if (wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA1) {
    const uint32_t causes = rocjitsu::cdna1::classify_binary_f32<true>(inst, wf, inst.vsrc1);
    wf.raise_alu_causes(causes);
    return causes;
  }
  return detail::classify_mul_lanes(
      inst.src0, inst.vsrc1, wf, [](float lhs, float rhs) { return classify_mul_f32(lhs, rhs); });
}

template <typename Inst> uint32_t classify_mul_f32_vop3(const Inst &inst, Wavefront &wf) {
  if (wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA3_5) {
    const uint32_t causes =
        rdna3_5::arithmetic_flags<rdna3_5::FmaOperation::Multiply>(inst, wf, inst.src1);
    wf.raise_alu_causes(causes);
    return causes;
  }
  if (wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA1) {
    const uint32_t causes = rocjitsu::cdna1::classify_binary_f32<true>(inst, wf, inst.src1);
    wf.raise_alu_causes(causes);
    return causes;
  }
  // OMOD scales the exact and rounded products alike. Its policy is uniform
  // across the wave, including which causes the target suppresses.
  const uint32_t omod = fp_mode::effective_omod(wf.cu().arch(), wf.fp_denorm_mode_f32(),
                                                wf.ieee_mode(), inst.inst_.omod);
  const float omod_scale = omod == 1 ? 2.0f : omod == 2 ? 4.0f : omod == 3 ? 0.5f : 1.0f;
  const uint32_t suppressed = omod != 0 && (wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA4 ||
                                            wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA5)
                                  ? (1u << 4) | (1u << 5)
                                  : 0;
  return detail::classify_mul_lanes(inst.src0, inst.src1, wf, [&](float lhs, float rhs) {
    if (inst.inst_.abs & 1u)
      lhs = std::fabs(lhs);
    if (inst.inst_.neg & 1u)
      lhs = -lhs;
    if (inst.inst_.abs & 2u)
      rhs = std::fabs(rhs);
    if (inst.inst_.neg & 2u)
      rhs = -rhs;
    return classify_mul_f32(lhs, rhs, omod_scale) & ~suppressed;
  });
}

// OMOD scales the rounded half. Report exceptions at the arithmetic and
// output boundaries before destination saturation or CLAMP masks the result.
inline uint32_t gfx10_f16_output_causes(double exact, uint16_t arithmetic, bool saturate,
                                        unsigned omod) {
  uint16_t magnitude = arithmetic & 0x7fffu;
  if (magnitude == 0 && exact != 0)
    return (1u << 4) | (1u << 5);
  if (magnitude == 0x7c00u && saturate)
    magnitude = 0x7bffu;
  const unsigned exponent = magnitude >> 10;
  if (exponent != 0 && exponent != 31) {
    if ((omod == 1 || omod == 2) && exponent + omod >= 31)
      return (1u << 3) | (1u << 5);
    if (omod == 3 && exponent == 1)
      return (1u << 4) | (1u << 5);
  }
  return 0;
}

template <typename Inst> void record_gfx10_add_f16_exceptions(const Inst &inst, Wavefront &wf) {
  const RegisterAccess regs(wf);
  const uint64_t active = wf.exec() & wf.vgpr_write_mask();
  const auto denorm = wf.fp_denorm_mode_f16_f64();
  const auto round_mode = wf.fp_round_mode_f16_f64();
  unsigned omod = 0;
  if constexpr (requires { inst.inst_.omod; })
    omod =
        fp_mode::effective_f16_omod(wf.cu().arch(), denorm, wf.ieee_mode(), false, inst.inst_.omod);
  else if constexpr (requires { inst.sdwa_omod_; })
    if (inst.inst_.src0 == SRC_SDWA)
      omod = fp_mode::effective_f16_omod(wf.cu().arch(), denorm, wf.ieee_mode(), false,
                                         inst.sdwa_omod_);
  uint32_t causes = 0;
  for (unsigned lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(active & (uint64_t{1} << lane)))
      continue;
    uint16_t a = uint16_t(regs.read_lane(inst.src0, lane));
    uint16_t b;
    if constexpr (requires { inst.src1; }) {
      b = uint16_t(regs.read_lane(inst.src1, lane));
      a = fp_mode::detail::modify_f16(a, inst.inst_.abs & 1, inst.inst_.neg & 1);
      b = fp_mode::detail::modify_f16(b, inst.inst_.abs & 2, inst.inst_.neg & 2);
    } else {
      b = uint16_t(regs.read_lane(inst.vsrc1, lane));
    }
    const auto ma = a & 0x7fffu, mb = b & 0x7fffu;
    if (ma >= 0x7c00 || mb >= 0x7c00) {
      if ((ma > 0x7c00 && !(a & 0x200)) || (mb > 0x7c00 && !(b & 0x200)) ||
          (ma == 0x7c00 && mb == 0x7c00 && a != b))
        causes |= 1u;
      continue;
    }
    if ((denorm & 1) && ((ma && ma < 0x400) || (mb && mb < 0x400)))
      causes |= 1u << 1;
    a = fp_mode::detail::flush_input_f16(a, denorm);
    b = fp_mode::detail::flush_input_f16(b, denorm);
    // The add rounds and flushes to a half before OMOD sees its result.
    // Detect arithmetic overflow independently of final FP16_OVFL saturation.
    const double exact = double(util::f16_to_f32(a)) + double(util::f16_to_f32(b));
    const auto raw = fp_mode::finish_arithmetic_f16(exact, round_mode, denorm, false);
    if ((raw & 0x7fffu) == 0x7c00 || std::abs(exact) >= 65536.0)
      causes |= 1u << 3;
    if (double(util::f16_to_f32(raw)) != exact) {
      causes |= 1u << 5;
      if (exact != 0 && std::abs(exact) < 0x1p-14)
        causes |= 1u << 4;
    }
    causes |= gfx10_f16_output_causes(exact, raw, wf.fp16_ovfl(), omod);
  }
  wf.set_trapsts(wf.trapsts() | causes);
  wf.raise_alu_causes(causes);
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

/// @brief Classify FIXUP causes before output clamping.
/// Input flushing suppresses input-denormal causes. Recovery branches also report
/// underflow/overflow and inexact, as measured on gfx1201; OMOD suppresses
/// underflow and inexact, but retains overflow.
template <typename Float>
uint32_t classify_div_fixup(Float quotient, Float denominator, Float numerator, uint32_t rounding,
                            uint32_t denorm, uint32_t omod) {
  using F = DivisionFormat<Float>;
  using Bits = typename F::Bits;
  Bits d = std::bit_cast<Bits>(denominator) & ~F::sign;
  Bits n = std::bit_cast<Bits>(numerator) & ~F::sign;
  const bool d_subnormal = d != 0 && F::exponent(d) == 0;
  const bool n_subnormal = n != 0 && F::exponent(n) == 0;
  uint32_t causes = (denorm & 1u) && (d_subnormal || n_subnormal) ? 1u << 1 : 0;
  if (!(denorm & 1u)) {
    if (d_subnormal)
      d = 0;
    if (n_subnormal)
      n = 0;
  }
  if ((d > F::infinity && !(d & F::quiet)) || (n > F::infinity && !(n & F::quiet)) ||
      (d == 0 && n == 0) || (d == F::infinity && n == F::infinity))
    causes |= 1u;
  else if (d == 0 && n != 0 && n < F::infinity)
    causes |= 1u << 2;
  if (d != 0 && d < F::infinity && n != 0 && n < F::infinity) {
    if (F::exponent(n) - F::exponent(d) < -(F::bias + F::fraction))
      causes |= (1u << 4) | (1u << 5);
    else if ((std::bit_cast<Bits>(quotient) & ~F::sign) >= F::infinity)
      causes |= (1u << 3) | (1u << 5);
  }
  if (omod != 0)
    causes &= ~((1u << 4) | (1u << 5));
  // OMOD can overflow a finite quotient. Classify before CLAMP changes the
  // numeric result, and only when the architecture enables the encoded OMOD.
  if (omod == 1 || omod == 2) {
    const Float result = div_fixup(quotient, denominator, numerator, rounding, denorm);
    const Bits bits = std::bit_cast<Bits>(result) & ~F::sign;
    if (bits < F::infinity && F::exponent(bits) + static_cast<int>(omod) >= 2 * F::bias + 1)
      causes |= 1u << 3;
  }
  return causes;
}

template <typename Float, typename Inst>
uint32_t classify_div_fixup_exceptions(const Inst &inst, Wavefront &wf) {
  static_assert(std::is_same_v<Float, float> || std::is_same_v<Float, double>);
  const uint64_t exec = wf.exec();
  const uint32_t denorm =
      std::is_same_v<Float, float> ? wf.fp_denorm_mode_f32() : wf.fp_denorm_mode_f16_f64();
  const uint32_t rounding =
      std::is_same_v<Float, float> ? wf.fp_round_mode_f32() : wf.fp_round_mode_f16_f64();
  const uint32_t omod =
      fp_mode::effective_omod(wf.cu().arch(), denorm, wf.ieee_mode(), inst.inst_.omod);
  uint32_t causes = 0;
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    Float q, d, n;
    if constexpr (std::is_same_v<Float, float>) {
      q = std::bit_cast<float>(RegisterAccess(wf).read_lane(inst.src0, lane));
      d = std::bit_cast<float>(RegisterAccess(wf).read_lane(inst.src1, lane));
      n = std::bit_cast<float>(RegisterAccess(wf).read_lane(inst.src2, lane));
    } else {
      q = std::bit_cast<double>(RegisterAccess(wf).read_lane64(inst.src0, lane));
      d = std::bit_cast<double>(RegisterAccess(wf).read_lane64(inst.src1, lane));
      n = std::bit_cast<double>(RegisterAccess(wf).read_lane64(inst.src2, lane));
    }
    // Signs affect directed-rounding overflow recovery before OMOD is applied.
    if (inst.inst_.abs & 1u)
      q = std::fabs(q);
    if (inst.inst_.neg & 1u)
      q = -q;
    if (inst.inst_.abs & 2u)
      d = std::fabs(d);
    if (inst.inst_.neg & 2u)
      d = -d;
    if (inst.inst_.abs & 4u)
      n = std::fabs(n);
    if (inst.inst_.neg & 4u)
      n = -n;
    causes |= classify_div_fixup(q, d, n, rounding, denorm, omod);
  }
  wf.raise_alu_causes(causes);
  return causes;
}

template <typename Inst>
uint32_t classify_div_fixup_f32_exceptions(const Inst &inst, Wavefront &wf) {
  return classify_div_fixup_exceptions<float>(inst, wf);
}

template <typename Inst>
uint32_t classify_div_fixup_f64_exceptions(const Inst &inst, Wavefront &wf) {
  return classify_div_fixup_exceptions<double>(inst, wf);
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

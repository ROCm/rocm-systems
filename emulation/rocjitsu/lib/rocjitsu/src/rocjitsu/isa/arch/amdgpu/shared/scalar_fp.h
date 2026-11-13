// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/comparison.h"
#include "rocjitsu/isa/arch/amdgpu/shared/conversion.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_status.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_f16_source.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "util/simd.h"

namespace rocjitsu::amdgpu::scalar_fp {

using F16 = amdgpu::fp_format::F16;
using F32 = amdgpu::fp_format::F32;
using I32 = amdgpu::conversion::I32;
using U32 = amdgpu::conversion::U32;
using amdgpu::RegisterAccess;
using amdgpu::Wavefront;
namespace fp = amdgpu::fp_mode;
namespace cvt = amdgpu::conversion;
namespace status = amdgpu::fp_status;

enum class Op { Add, Sub, Mul, Fma, Ceil, Floor, Trunc, Rndne };

inline bool use_ieee_nan(const Wavefront &wf) {
  return wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA4 || wf.ieee_mode();
}

template <typename Fmt> uint32_t denorm(const Wavefront &wf) {
  if constexpr (std::is_same_v<Fmt, F16>)
    return wf.fp_denorm_mode_f16_f64();
  return wf.fp_denorm_mode_f32();
}
template <typename Fmt> uint32_t rounding(const Wavefront &wf) {
  if constexpr (std::is_same_v<Fmt, F16>)
    return wf.fp_round_mode_f16_f64();
  return wf.fp_round_mode_f32();
}

template <typename Fmt> uint32_t read(const Wavefront &wf, const Operand &op, bool high = false) {
  if constexpr (std::is_same_v<Fmt, F16>)
    return (amdgpu::read_scalar_f16_carrier(op, wf, high) >> (high ? 16 : 0)) & 0xffffu;
  return RegisterAccess(wf).read_scalar(op);
}

template <typename Fmt> uint32_t prepare(const Wavefront &wf, uint32_t bits) {
  return amdgpu::input_denormal::prepare<Fmt>(
      bits, amdgpu::input_denormal::Policy::make(denorm<Fmt>(wf)));
}

inline void commit_causes(Wavefront &wf, uint32_t causes) {
  if (!wf.instruction_execution_failed() &&
      (wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA4 || !wf.dx10_clamp())) {
    wf.set_trapsts(wf.trapsts() | causes);
    wf.raise_alu_causes(causes);
  }
}

inline void commit(Wavefront &wf, const Operand &dst, uint32_t result, uint32_t causes) {
  if (!wf.instruction_execution_failed()) {
    RegisterAccess(wf).write_scalar(dst, result);
    commit_causes(wf, causes);
  }
}

// Scalar comparisons and min/max suppress input-denormal status when either operand is nonfinite.
template <typename Fmt> uint32_t binary_input_causes(uint32_t a, uint32_t b, uint32_t denorm) {
  if ((a & Fmt::kMagnitude) >= Fmt::kInfinity || (b & Fmt::kMagnitude) >= Fmt::kInfinity)
    denorm = 0;
  return status::input_causes<Fmt>(a, denorm) | status::input_causes<Fmt>(b, denorm);
}

// Lower ADD/SUB/MUL to the same target FMA used by the vector pipeline.
// Multiplication adds the product's signed zero. Addition multiplies by one.
template <typename Fmt, Op op>
void arithmetic(Wavefront &wf, const Operand &dst, const Operand &lhs, const Operand &rhs,
                const Operand *acc = nullptr) {
  uint32_t multiplicand_bits = read<Fmt>(wf, lhs), multiplier_bits = read<Fmt>(wf, rhs);
  uint32_t addend_bits = acc ? read<Fmt>(wf, *acc) : 0;
  if constexpr (op == Op::Add || op == Op::Sub) {
    addend_bits = op == Op::Sub ? multiplier_bits ^ Fmt::kSign : multiplier_bits;
    multiplier_bits = Fmt::kBias << Fmt::kMantissaBits;
  } else if constexpr (op == Op::Mul) {
    addend_bits = (multiplicand_bits ^ multiplier_bits) & Fmt::kSign;
  }
  uint32_t result;
  const uint32_t round = rounding<Fmt>(wf), flush = denorm<Fmt>(wf);
  if constexpr (std::is_same_v<Fmt, F16>) {
    result =
        fp::fma_f16(multiplicand_bits, multiplier_bits, addend_bits, false, false, false, false,
                    false, false, round, flush, 0, false, wf.fp16_ovfl(), false, use_ieee_nan(wf));
  } else {
    result = std::bit_cast<uint32_t>(fp::arithmetic<fp::Arithmetic::FMA>(
        std::bit_cast<float>(multiplicand_bits), std::bit_cast<float>(multiplier_bits),
        std::bit_cast<float>(addend_bits), round, flush, wf.cu().arch(), wf.ieee_mode()));
  }
  uint32_t causes =
      status::fma<Fmt>(multiplicand_bits, multiplier_bits, addend_bits, result, round, flush);
  // Scalar arithmetic preserves input-denormal status for an infinite result.
  // A NaN result, including an invalid fused operation, suppresses it.
  if (amdgpu::comparison::is_nan<Fmt>(result))
    causes &= ~status::kInputDenorm;
  commit(wf, dst, result, causes);
}

template <typename Fmt, typename Rel>
void compare(Wavefront &wf, const Operand &lhs, const Operand &rhs) {
  const uint32_t a = read<Fmt>(wf, lhs), b = read<Fmt>(wf, rhs);
  if (wf.instruction_execution_failed())
    return;
  const uint32_t flush = denorm<Fmt>(wf);
  wf.write_scc(
      amdgpu::comparison::evaluate<Fmt, Rel>(a, b, amdgpu::input_denormal::Policy::make(flush)));
  commit_causes(wf, binary_input_causes<Fmt>(a, b, flush));
}

template <typename Fmt, typename Which>
void select(Wavefront &wf, const Operand &dst, const Operand &lhs, const Operand &rhs) {
  const uint32_t a = read<Fmt>(wf, lhs), b = read<Fmt>(wf, rhs), flush = denorm<Fmt>(wf);
  const auto input = amdgpu::input_denormal::Policy::make(flush);
  const auto result = [&] {
    if constexpr (amdgpu::minmax::detail::is_legacy_v<Which>)
      return amdgpu::minmax::LegacyOperation<Fmt, Which>{
          input, fp::legacy_minmax_policy(wf.cu().arch(), wf.ieee_mode())}(a, b);
    else
      return amdgpu::minmax::evaluate<Fmt, Which>(input, a, b);
  }();
  commit(wf, dst, result, binary_input_causes<Fmt>(a, b, flush));
}

template <typename Fmt, Op op>
void round_integral(Wavefront &wf, const Operand &dst, const Operand &source) {
  const uint32_t input_bits = prepare<Fmt>(wf, read<Fmt>(wf, source));
  uint32_t result = input_bits;
  if (amdgpu::comparison::is_nan<Fmt>(input_bits)) {
    if (use_ieee_nan(wf))
      result |= Fmt::kQuiet;
  } else {
    fp::ScopedEnvironment environment(0);
    const float input = static_cast<float>(status::value<Fmt>(input_bits));
    float rounded;
    if constexpr (op == Op::Ceil)
      rounded = std::ceil(input);
    else if constexpr (op == Op::Floor)
      rounded = std::floor(input);
    else if constexpr (op == Op::Trunc)
      rounded = std::trunc(input);
    else
      rounded = util::rndne_scalar(input);
    if constexpr (std::is_same_v<Fmt, F16>)
      result = util::f32_to_f16(rounded);
    else
      result = std::bit_cast<uint32_t>(rounded);
  }
  commit(wf, dst, result, status::input_causes<Fmt>(input_bits, denorm<Fmt>(wf)));
}

template <typename From, typename To>
uint32_t to_float(Wavefront &wf, uint32_t a, uint32_t round, uint32_t &causes) {
  cvt::ToFloat<From, To> policy;
  if constexpr (!cvt::is_integer_v<From>) {
    policy.input = amdgpu::input_denormal::Policy::make(denorm<From>(wf));
    causes |= status::input_causes<From>(a, denorm<From>(wf));
  }
  policy.rounding.direction = cvt::mode_rounding(round);
  policy.rounding.flush_tiny = !(denorm<To>(wf) & 2);
  policy.rounding.saturate = std::is_same_v<To, F16> && wf.fp16_ovfl();
  policy.quiet_nan = use_ieee_nan(wf);
  const uint32_t result = policy(a);
  double exact;
  if constexpr (std::is_same_v<From, I32>)
    exact = static_cast<int32_t>(a);
  else if constexpr (std::is_same_v<From, U32>)
    exact = a;
  else
    exact = status::value<From>(prepare<From>(wf, a));
  if (std::isfinite(exact))
    causes |= status::result_causes<To>(exact, result, round);
  return result;
}

template <typename From, typename To>
void convert(Wavefront &wf, const Operand &dst, const Operand &src, bool high = false) {
  const uint32_t a = read<From>(wf, src, high);
  uint32_t causes = 0, result;
  {
    fp::ScopedEnvironment environment(0);
    if constexpr (cvt::is_integer_v<To>) {
      using Int = std::conditional_t<To::kSigned, int32_t, uint32_t>;
      const uint32_t bits = prepare<From>(wf, a);
      result = cvt::f32_to_integer<Int>(bits, cvt::Rounding::TOWARD_ZERO, cvt::NanResult::ZERO);
      causes = status::input_causes<From>(a, denorm<From>(wf));
      const double x = status::value<From>(bits), truncated = std::trunc(x);
      if (!std::isfinite(x) || x < std::numeric_limits<Int>::min() ||
          truncated > std::numeric_limits<Int>::max())
        causes = status::kInvalid;
      else if (wf.cu().arch() != ROCJITSU_CODE_ARCH_RDNA4 && truncated != x)
        causes |= status::kInexact;
    } else {
      result = to_float<From, To>(wf, a, rounding<To>(wf), causes);
    }
  }
  commit(wf, dst, result, causes);
}

inline void pack_rtz(Wavefront &wf, const Operand &dst, const Operand &lo, const Operand &hi) {
  const uint32_t low_input = read<F32>(wf, lo);
  const uint32_t high_input = read<F32>(wf, hi);
  uint32_t low, high, causes = 0;
  {
    fp::ScopedEnvironment environment(0);
    if (wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA4) {
      // gfx1201 packs both conversions without changing floating-point status.
      cvt::ToFloat<F32, F16> policy;
      policy.input = amdgpu::input_denormal::Policy::make(denorm<F32>(wf));
      policy.rounding.direction = cvt::Rounding::TOWARD_ZERO;
      policy.rounding.flush_tiny = !(denorm<F16>(wf) & 2);
      policy.quiet_nan = true;
      low = policy(low_input);
      high = policy(high_input);
    } else {
      low = to_float<F32, F16>(wf, low_input, 3, causes);
      high = to_float<F32, F16>(wf, high_input, 3, causes);
    }
  }
  commit(wf, dst, low | (high << 16), causes);
}
} // namespace rocjitsu::amdgpu::scalar_fp

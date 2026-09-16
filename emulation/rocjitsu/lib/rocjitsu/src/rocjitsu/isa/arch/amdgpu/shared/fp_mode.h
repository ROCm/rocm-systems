// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file fp_mode.h
/// @brief Shared MODE-aware floating-point execution helpers.

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/shared/pseudo_scalar.h"
#include "util/data_types.h"

#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <xmmintrin.h>
#endif

namespace rocjitsu::amdgpu::fp_mode {

namespace detail {

inline uint16_t modify_f16(uint16_t value, bool absolute, bool negate) {
  if (absolute)
    value &= 0x7fffu;
  if (negate)
    value ^= 0x8000u;
  return value;
}

inline uint16_t flush_input_f16(uint16_t value, uint32_t denorm_mode) {
  if ((denorm_mode & 1u) == 0 && (value & 0x7c00u) == 0 && (value & 0x03ffu) != 0)
    return value & 0x8000u;
  return value;
}

inline uint64_t flush_f64(uint64_t value) {
  if ((value & 0x7ff0000000000000ULL) == 0 && (value & 0x000fffffffffffffULL) != 0)
    return value & 0x8000000000000000ULL;
  return value;
}

inline int host_round_mode(uint32_t round_mode) {
  switch (round_mode & 3u) {
  case 1:
    return FE_UPWARD;
  case 2:
    return FE_DOWNWARD;
  case 3:
    return FE_TOWARDZERO;
  default:
    return FE_TONEAREST;
  }
}

#if defined(__aarch64__)
inline uint64_t read_fpcr() {
  uint64_t value;
  __asm__ volatile("mrs %0, fpcr" : "=r"(value) : : "memory");
  return value;
}

inline void write_fpcr(uint64_t value) {
  __asm__ volatile("msr fpcr, %0" : : "r"(value) : "memory");
}
#endif

class ScopedFenv {
public:
  explicit ScopedFenv(uint32_t round_mode) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    saved_mxcsr_ = _mm_getcsr();
#elif defined(__aarch64__)
    saved_fpcr_ = read_fpcr();
#endif
    saved_ = std::feholdexcept(&environment_) == 0;
    if (saved_)
      std::fesetround(host_round_mode(round_mode));
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    constexpr uint32_t kDazMask = 1u << 6;
    constexpr uint32_t kFtzMask = 1u << 15;
    _mm_setcsr(_mm_getcsr() & ~(kDazMask | kFtzMask));
#elif defined(__aarch64__)
    constexpr uint64_t kFizMask = uint64_t{1} << 0;
    constexpr uint64_t kFz16Mask = uint64_t{1} << 19;
    constexpr uint64_t kFzMask = uint64_t{1} << 24;
    write_fpcr(read_fpcr() & ~(kFizMask | kFz16Mask | kFzMask));
#endif
  }

  ScopedFenv(const ScopedFenv &) = delete;
  ScopedFenv &operator=(const ScopedFenv &) = delete;

  ~ScopedFenv() {
    if (saved_)
      std::fesetenv(&environment_);
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    _mm_setcsr(saved_mxcsr_);
#elif defined(__aarch64__)
    write_fpcr(saved_fpcr_);
#endif
  }

private:
  std::fenv_t environment_{};
  bool saved_ = false;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  uint32_t saved_mxcsr_ = 0;
#elif defined(__aarch64__)
  uint64_t saved_fpcr_ = 0;
#endif
};

struct ExactF64Sum {
  double value;
  double error;
};

/// @brief Add two finite F64 values while retaining the exact rounding residual.
inline ExactF64Sum add_exact(double lhs, double rhs) {
  const double value = lhs + rhs;
  const double rhs_virtual = value - lhs;
  const double error = (lhs - (value - rhs_virtual)) + (rhs - rhs_virtual);
  return {value, error};
}

/// @brief Compare an exact two-component sum with an exactly representable value.
inline int compare_exact(ExactF64Sum sum, double value) {
  if (sum.value < value)
    return -1;
  if (sum.value > value)
    return 1;
  return sum.error < 0.0 ? -1 : sum.error > 0.0 ? 1 : 0;
}

/// @brief Convert F32 to BF16 using an AMDGPU MODE.FP_ROUND encoding.
inline uint16_t f32_to_bf16_round(float value, uint32_t round_mode) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  if ((round_mode & 3u) == 0u || (bits & 0x7f800000u) == 0x7f800000u)
    return util::f32_to_bf16_rne(value);

  uint16_t result = static_cast<uint16_t>(bits >> 16);
  if ((bits & 0xffffu) == 0u)
    return result;

  const bool negative = (bits & 0x80000000u) != 0u;
  if (((round_mode & 3u) == 1u && !negative) || ((round_mode & 3u) == 2u && negative))
    ++result;
  return result;
}

inline uint16_t next_up_bf16(uint16_t value) {
  if ((value & 0x7fffu) > 0x7f80u || value == 0x7f80u)
    return value;
  if (value == 0xff80u)
    return 0xff7fu;
  if ((value & 0x7fffu) == 0)
    return 0x0001u;
  return static_cast<uint16_t>((value & 0x8000u) != 0 ? value - 1u : value + 1u);
}

inline uint16_t next_down_bf16(uint16_t value) {
  if ((value & 0x7fffu) > 0x7f80u || value == 0xff80u)
    return value;
  if (value == 0x7f80u)
    return 0x7f7fu;
  if ((value & 0x7fffu) == 0)
    return 0x8001u;
  return static_cast<uint16_t>((value & 0x8000u) != 0 ? value + 1u : value - 1u);
}

inline uint16_t round_exact_to_bf16(ExactF64Sum sum, uint32_t round_mode) {
  constexpr double kMaxBf16 = 0x1.fep127;
  constexpr double kRneOverflowThreshold = 0x1.ffp127;
  const int zero_cmp = compare_exact(sum, 0.0);

  if (compare_exact(sum, kMaxBf16) > 0) {
    if ((round_mode & 3u) == 1u ||
        ((round_mode & 3u) == 0u && compare_exact(sum, kRneOverflowThreshold) >= 0))
      return 0x7f80u;
    return 0x7f7fu;
  }
  if (compare_exact(sum, -kMaxBf16) < 0) {
    if ((round_mode & 3u) == 2u ||
        ((round_mode & 3u) == 0u && compare_exact(sum, -kRneOverflowThreshold) <= 0))
      return 0xff80u;
    return 0xff7fu;
  }

  const float approximation = static_cast<float>(sum.value);
  const uint16_t candidate = util::f32_to_bf16_rne(approximation);
  const double candidate_value = util::bf16_to_f32(candidate);
  const int candidate_cmp = compare_exact(sum, candidate_value);
  if (candidate_cmp == 0)
    return candidate;

  const uint16_t lower = candidate_cmp > 0 ? candidate : next_down_bf16(candidate);
  const uint16_t upper = candidate_cmp < 0 ? candidate : next_up_bf16(candidate);
  switch (round_mode & 3u) {
  case 0: {
    const double midpoint =
        (static_cast<double>(util::bf16_to_f32(lower)) + util::bf16_to_f32(upper)) * 0.5;
    const int midpoint_cmp = compare_exact(sum, midpoint);
    if (midpoint_cmp < 0)
      return lower;
    if (midpoint_cmp > 0)
      return upper;
    return (lower & 1u) == 0 ? lower : upper;
  }
  case 1:
    return upper;
  case 2:
    return lower;
  case 3:
    return zero_cmp < 0 ? upper : lower;
  default:
    return candidate;
  }
}

} // namespace detail

/// @brief Return the OMOD value supported by an ordinary floating-point result.
inline uint32_t effective_omod(rj_code_arch_t arch, uint32_t denorm_mode, bool ieee_mode,
                               uint32_t omod) {
  if (omod == 0)
    return 0;
  if (arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5)
    return omod;
  return (denorm_mode & 2u) == 0 && !ieee_mode ? omod : 0;
}

/// @brief Return the OMOD value that applies to an F16 result on the selected ISA.
/// @details GFX11+ packed-F16 results explicitly ignore OMOD. Older profiles expose
/// OMOD on the promoted VOP3 form of V_PK_FMAC_F16, subject to their ordinary
/// output-denormal and MODE.IEEE restrictions. GFX12 and gfx1250 allow OMOD on
/// non-packed F16 results regardless of output-denormal mode.
inline uint32_t effective_f16_omod(rj_code_arch_t arch, uint32_t denorm_mode, bool ieee_mode,
                                   bool packed_result, uint32_t omod) {
  if (omod == 0)
    return 0;
  if (packed_result && (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
                        arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5))
    return 0;
  return effective_omod(arch, denorm_mode, ieee_mode, omod);
}

/// @brief Apply the result-format rules required by an active OMOD.
/// @details OMOD always flushes an output subnormal and maps either signed zero
/// to positive zero. These helpers operate after the result has been rounded to
/// its architectural destination format.
inline uint16_t finalize_omod_f16(uint16_t value, uint32_t omod) {
  if (omod == 0)
    return value;
  if ((value & 0x7c00u) == 0 && (value & 0x03ffu) != 0)
    value &= 0x8000u;
  return (value & 0x7fffu) == 0 ? 0 : value;
}

inline uint16_t finalize_omod_bf16(uint16_t value, uint32_t omod) {
  if (omod == 0)
    return value;
  if ((value & 0x7f80u) == 0 && (value & 0x007fu) != 0)
    value &= 0x8000u;
  return (value & 0x7fffu) == 0 ? 0 : value;
}

namespace detail {

/// @brief Execute F32-source fused multiply-add in an established clean nearest environment.
inline uint16_t fma_f32_to_bf16_nearest_environment(float multiplicand, float multiplier,
                                                    float addend, uint32_t round_mode, bool clamp,
                                                    bool clamp_nan_to_zero) {
  if (!std::isfinite(multiplicand) || !std::isfinite(multiplier) || !std::isfinite(addend)) {
    float result = std::fma(multiplicand, multiplier, addend);
    if (clamp) {
      if ((clamp_nan_to_zero && std::isnan(result)) || result <= 0.0f)
        result = 0.0f;
      else if (result > 1.0f)
        result = 1.0f;
    }
    return detail::f32_to_bf16_round(result, round_mode);
  }

  const double product = static_cast<double>(multiplicand) * static_cast<double>(multiplier);
  const detail::ExactF64Sum exact = detail::add_exact(product, static_cast<double>(addend));
  if (exact.value == 0.0 && exact.error == 0.0) {
    if (clamp)
      return 0;
    detail::ScopedFenv result_environment(round_mode);
    return detail::f32_to_bf16_round(std::fma(multiplicand, multiplier, addend), round_mode);
  }
  if (clamp) {
    if (detail::compare_exact(exact, 0.0) <= 0)
      return 0;
    if (detail::compare_exact(exact, 1.0) > 0)
      return 0x3f80u;
  }
  return detail::round_exact_to_bf16(exact, round_mode);
}

} // namespace detail

/// @brief Execute an F32-source fused multiply-add and round once to BF16.
/// @details An F32 product is exact in F64. The error-free sum retains the exact addend residual,
/// which is needed when the rounded F64 value lands on a BF16 boundary or midpoint.
inline uint16_t fma_f32_to_bf16(float multiplicand, float multiplier, float addend,
                                uint32_t round_mode, bool clamp, bool clamp_nan_to_zero) {
  detail::ScopedFenv nearest_environment(0);
  return detail::fma_f32_to_bf16_nearest_environment(multiplicand, multiplier, addend, round_mode,
                                                     clamp, clamp_nan_to_zero);
}

/// @brief Isolate scalar or SIMD arithmetic from the caller's rounding and flush controls.
using ScopedEnvironment = detail::ScopedFenv;

enum class PackedF32Op { ADD, MUL, FMA };

enum class PackedBinaryOp { ADD, MUL, MIN, MAX, MINIMUM, MAXIMUM };

/// @brief Packed F16 binary arithmetic uses the FP16/64 MODE controls.
inline uint16_t packed_binary_f16(PackedBinaryOp operation, uint16_t a_bits, uint16_t b_bits,
                                  uint32_t round_mode, uint32_t denorm_mode, bool clamp,
                                  bool fp16_ovfl, bool clamp_nan_to_zero) {
  detail::ScopedFenv nearest_environment(0);
  a_bits = detail::flush_input_f16(a_bits, denorm_mode);
  b_bits = detail::flush_input_f16(b_bits, denorm_mode);
  const double first = util::f16_to_f32(a_bits);
  const double second = util::f16_to_f32(b_bits);
  double value = std::numeric_limits<double>::quiet_NaN();
  switch (operation) {
  case PackedBinaryOp::ADD:
    value = first + second;
    // Exact cancellation follows the destination's directed rounding mode.
    if (value == 0.0 && std::signbit(first) != std::signbit(second))
      value = round_mode == 2 ? -0.0 : 0.0;
    break;
  case PackedBinaryOp::MUL:
    value = first * second;
    break;
  case PackedBinaryOp::MIN:
  case PackedBinaryOp::MAX:
  case PackedBinaryOp::MINIMUM:
  case PackedBinaryOp::MAXIMUM: {
    const bool minimum = operation == PackedBinaryOp::MIN || operation == PackedBinaryOp::MINIMUM;
    const bool propagate_nan =
        operation == PackedBinaryOp::MINIMUM || operation == PackedBinaryOp::MAXIMUM;
    if (propagate_nan && (std::isnan(first) || std::isnan(second)))
      value = std::numeric_limits<double>::quiet_NaN();
    else if (first == second)
      value =
          minimum ? (std::signbit(first) ? first : second) : (std::signbit(first) ? second : first);
    else
      value = minimum ? std::fmin(first, second) : std::fmax(first, second);
    break;
  }
  }
  uint16_t result =
      pseudo_scalar::round_f16_result(value, round_mode, 0, clamp, fp16_ovfl, clamp_nan_to_zero);
  if (!(denorm_mode & 2u) && (result & 0x7c00u) == 0)
    result &= 0x8000u;
  return result;
}

/// @brief Select three packed F16 operands, applying output controls only to the final result.
inline uint16_t packed_select3_f16(PackedBinaryOp operation, uint16_t first, uint16_t second,
                                   uint16_t third, uint32_t denorm_mode, bool clamp,
                                   bool clamp_nan_to_zero) {
  const uint16_t pair =
      packed_binary_f16(operation, first, second, 0, denorm_mode | 2u, false, false, false);
  return packed_binary_f16(operation, pair, third, 0, denorm_mode, clamp, false, clamp_nan_to_zero);
}

/// @brief Packed BF16 numeric min/max preserve denormals and order signed zeros explicitly.
inline uint16_t packed_select_bf16(float first, float second, bool minimum) {
  ScopedEnvironment environment(0);
  const float selected = first == second
                             ? (minimum ? (std::signbit(first) ? first : second)
                                        : (std::signbit(first) ? second : first))
                             : (minimum ? std::fmin(first, second) : std::fmax(first, second));
  return util::f32_to_bf16_rne(selected);
}

/// @brief Packed BF16 CLAMP is applied after the fixed-RNE operation.
inline uint16_t clamp_bf16(uint16_t value, bool clamp, bool clamp_nan_to_zero) {
  if (!clamp)
    return value;
  const uint16_t magnitude = value & 0x7fffu;
  if (magnitude > 0x7f80u)
    return clamp_nan_to_zero ? 0 : value;
  if (value & 0x8000u)
    return 0;
  return value > 0x3f80u ? 0x3f80u : value;
}

/// @brief Packed F32 operations use FP32 MODE, independently of the host environment.
inline uint32_t packed_f32(float first, float second, float third, PackedF32Op operation,
                           uint32_t round_mode, uint32_t denorm_mode, bool clamp,
                           bool clamp_nan_to_zero) {
  detail::ScopedFenv environment(round_mode);
  auto flush = [](float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    return (bits & 0x7f800000u) == 0 ? std::bit_cast<float>(bits & 0x80000000u) : value;
  };
  volatile float first_input = (denorm_mode & 1u) ? first : flush(first);
  volatile float second_input = (denorm_mode & 1u) ? second : flush(second);
  volatile float third_input = (denorm_mode & 1u) ? third : flush(third);
  float result = std::numeric_limits<float>::quiet_NaN();
  switch (operation) {
  case PackedF32Op::ADD:
    result = first_input + second_input;
    break;
  case PackedF32Op::MUL:
    result = first_input * second_input;
    break;
  case PackedF32Op::FMA:
    result = std::fma(first_input, second_input, third_input);
    break;
  }
  if (clamp) {
    if (std::isnan(result)) {
      if (clamp_nan_to_zero)
        result = 0.0f;
    } else {
      result = result <= 0.0f ? 0.0f : result > 1.0f ? 1.0f : result;
    }
  }
  return std::bit_cast<uint32_t>((denorm_mode & 2u) ? result : flush(result));
}

/// @brief True16 F16 DOT2 uses fixed RNE without changing its F32 accumulation model.
inline uint16_t dot2_f16(float a0, float b0, float a1, float b1, float acc, bool fp16_ovfl) {
  detail::ScopedFenv nearest_environment(0);
  volatile float product0 = a0 * b0;
  volatile float product1 = a1 * b1;
  volatile float sum = product0 + product1;
  return util::f32_to_f16_mode(sum + acc, fp16_ovfl);
}

/// @brief Packed BF16 math always rounds once to nearest-even and preserves denormals.
inline uint16_t packed_fma_bf16(float a, float b, float c, bool fp16_ovfl) {
  uint16_t result = fma_f32_to_bf16(a, b, c, 0, false, false);
  // FP16_OVFL clamps finite-input results that round to BF16 infinity;
  // infinities supplied as operands retain their normal arithmetic behavior.
  if (fp16_ovfl && (result & 0x7fffu) == 0x7f80u && std::isfinite(a) && std::isfinite(b) &&
      std::isfinite(c))
    result = static_cast<uint16_t>((result & 0x8000u) | 0x7f7fu);
  return result;
}

inline uint16_t packed_add_bf16(float a, float b, bool fp16_ovfl) {
  return packed_fma_bf16(a, 1.0f, b, fp16_ovfl);
}

inline uint16_t packed_mul_bf16(float a, float b, bool fp16_ovfl) {
  // An addend with the product's sign preserves a negative zero product.
  const float zero = std::signbit(a) != std::signbit(b) ? -0.0f : 0.0f;
  return packed_fma_bf16(a, b, zero, fp16_ovfl);
}

enum class ScalarAtomicOp { FADD, FMIN, FMAX, FCMPSWAP };

/// @brief Execute scalar floating atomics with explicit ISA policy and bit-preserving selection.
template <typename Bits>
Bits atomic_scalar(ScalarAtomicOp operation, Bits old_bits, Bits source_bits, Bits compare_bits,
                   uint32_t denorm_mode, bool legacy_minmax) {
  static_assert(std::is_same_v<Bits, uint32_t> || std::is_same_v<Bits, uint64_t>);
  using Float = std::conditional_t<sizeof(Bits) == 4, float, double>;
  constexpr Bits kSign = Bits{1} << (sizeof(Bits) * 8 - 1);
  constexpr Bits kExponent =
      sizeof(Bits) == 4 ? Bits{0x7f800000u} : static_cast<Bits>(0x7ff0000000000000ULL);
  constexpr Bits kQuiet =
      sizeof(Bits) == 4 ? Bits{0x00400000u} : static_cast<Bits>(0x0008000000000000ULL);
  constexpr Bits kMantissa = ~(kSign | kExponent);
  auto flush = [](Bits bits) -> Bits { return (bits & kExponent) == 0 ? bits & kSign : bits; };
  auto is_nan = [](Bits bits) {
    return (bits & kExponent) == kExponent && (bits & kMantissa) != 0;
  };
  const Bits old_input = (denorm_mode & 1u) ? old_bits : flush(old_bits);
  const Bits source_input = (denorm_mode & 1u) ? source_bits : flush(source_bits);
  const Bits compare_input = (denorm_mode & 1u) ? compare_bits : flush(compare_bits);
  if (operation == ScalarAtomicOp::FCMPSWAP) {
    // Integer equality implements finite IEEE equality except for signed zero;
    // classify NaNs explicitly so equal payloads never cause a swap.
    const bool equal =
        old_input == compare_input || ((old_input & ~kSign) == 0 && (compare_input & ~kSign) == 0);
    return !is_nan(old_input) && !is_nan(compare_input) && equal ? source_input : old_input;
  }
  if (operation == ScalarAtomicOp::FADD) {
    // DS and cache ADD use fixed RNE, independent of both MODE.round and the host.
    // Select NaNs explicitly so host operand scheduling cannot change payload priority.
    if (is_nan(old_input))
      return old_input | kQuiet;
    if (is_nan(source_input))
      return source_input | kQuiet;
    if ((old_input & ~kSign) == kExponent && (source_input & ~kSign) == kExponent &&
        ((old_input ^ source_input) & kSign))
      return kSign | kExponent | kQuiet;
    detail::ScopedFenv nearest_environment(0);
    volatile Float old_value = std::bit_cast<Float>(old_input);
    volatile Float source_value = std::bit_cast<Float>(source_input);
    const Bits result = std::bit_cast<Bits>(static_cast<Float>(old_value + source_value));
    return (denorm_mode & 2u) ? result : flush(result);
  }

  // CDNA1-4 / RDNA1-3.5 preserve selected denormals and propagate SNaNs.
  // CDNA5 / RDNA4 MIN_NUM/MAX_NUM select from the flushed values instead.
  const Bits old_result = legacy_minmax ? old_bits : old_input;
  const Bits source_result = legacy_minmax ? source_bits : source_input;
  if (legacy_minmax) {
    if (is_nan(old_input) && !(old_input & kQuiet))
      return old_input | kQuiet;
    if (is_nan(source_input) && !(source_input & kQuiet))
      return source_input | kQuiet;
  }
  if (is_nan(old_input))
    return is_nan(source_input) ? (legacy_minmax ? old_input : source_input) | kQuiet
                                : source_result;
  if (is_nan(source_input))
    return old_result;

  // Compare IEEE bit patterns directly. This also avoids host DAZ affecting a
  // preserved denormal comparison and orders -0 before +0 without host fmin/fmax.
  const Bits old_order = (old_input & kSign) ? ~old_input : old_input ^ kSign;
  const Bits source_order = (source_input & kSign) ? ~source_input : source_input ^ kSign;
  const bool select_source =
      operation == ScalarAtomicOp::FMIN ? source_order < old_order : source_order > old_order;
  return select_source ? source_result : old_result;
}

/// @brief Add packed F16/BF16 components using float-memory-atomic policy.
/// @details CDNA5 ISA chapter 12 fixes rounding to nearest-even and specifies
/// NaN selection. FP16_OVFL applies to VALU results, not memory atomics. The
/// caller supplies DS input/output denormal controls, or 3 for no flushing.
inline uint32_t atomic_add_packed_16(uint32_t old_val, uint32_t src_val, bool bf16,
                                     uint32_t denorm_mode) {
  detail::ScopedFenv nearest_environment(0);
  const uint16_t exponent_mask = bf16 ? 0x7f80u : 0x7c00u;
  auto flush_denorm = [&](uint16_t bits) -> uint16_t {
    return (bits & exponent_mask) == 0 ? bits & 0x8000u : bits;
  };
  uint32_t result = 0;
  for (uint32_t shift : {0u, 16u}) {
    uint16_t a = static_cast<uint16_t>(old_val >> shift);
    uint16_t b = static_cast<uint16_t>(src_val >> shift);
    if (!(denorm_mode & 1u)) {
      a = flush_denorm(a);
      b = flush_denorm(b);
    }
    const uint16_t quiet_bit = bf16 ? 0x0040u : 0x0200u;
    auto is_nan = [&](uint16_t bits) { return (bits & 0x7fffu) > exponent_mask; };
    uint16_t sum;
    // Float memory add propagates the first NaN, quieting it without changing
    // the sign or payload. Invalid infinity addition produces negative QNaN.
    if (is_nan(a))
      sum = a | quiet_bit;
    else if (is_nan(b))
      sum = b | quiet_bit;
    else if ((a & 0x7fffu) == exponent_mask && (b & 0x7fffu) == exponent_mask && (a ^ b) == 0x8000u)
      sum = 0x8000u | exponent_mask | quiet_bit;
    else
      sum = bf16 ? detail::fma_f32_to_bf16_nearest_environment(
                       util::bf16_to_f32(a), 1.0f, util::bf16_to_f32(b), 0, false, false)
                 : util::f32_to_f16(util::f16_to_f32(a) + util::f16_to_f32(b));
    if (!(denorm_mode & 2u))
      sum = flush_denorm(sum);
    result |= uint32_t{sum} << shift;
  }
  return result;
}

/// @brief Apply RDNA BF16 DOT2 rounding and denormal policy to the F32 evaluation.
/// @details RDNA3 section 7.2.4 (also used by RDNA3.5) and RDNA4 section 7.2.4 require
/// fixed RNE and flushed input/output denormals independently of MODE. This retains
/// the existing F32 accumulation, with RNE for its operations and final BF16 narrowing;
/// the accumulation precision and association are unchanged by this policy helper.
inline uint16_t dot2_bf16(float left_low, float right_low, float left_high, float right_high,
                          float accumulator) {
  detail::ScopedFenv nearest_environment(0);
  auto flush_input = [](float value) {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    if ((bits & 0x7f800000u) == 0)
      bits &= 0x80000000u;
    return std::bit_cast<float>(bits);
  };
  left_low = flush_input(left_low);
  right_low = flush_input(right_low);
  left_high = flush_input(left_high);
  right_high = flush_input(right_high);
  accumulator = flush_input(accumulator);
  const float result = left_low * right_low + left_high * right_high + accumulator;
  uint16_t result_bits = detail::f32_to_bf16_round(result, 0);
  if ((result_bits & 0x7f80u) == 0)
    result_bits &= 0x8000u;
  return result_bits;
}

inline float finalize_omod_f32(float value, uint32_t omod) {
  if (omod == 0)
    return value;
  uint32_t bits = std::bit_cast<uint32_t>(value);
  if ((bits & 0x7f800000u) == 0 && (bits & 0x007fffffu) != 0)
    bits &= 0x80000000u;
  if ((bits & 0x7fffffffu) == 0)
    bits = 0;
  return std::bit_cast<float>(bits);
}

inline double finalize_omod_f64(double value, uint32_t omod) {
  if (omod == 0)
    return value;
  uint64_t bits = detail::flush_f64(std::bit_cast<uint64_t>(value));
  if ((bits & 0x7fffffffffffffffULL) == 0)
    bits = 0;
  return std::bit_cast<double>(bits);
}

/// @brief Execute an F16 fused multiply-add and return its raw F16 encoding.
inline uint16_t fma_f16(uint16_t src0, uint16_t src1, uint16_t src2, bool abs0, bool abs1,
                        bool abs2, bool neg0, bool neg1, bool neg2, uint32_t round_mode,
                        uint32_t denorm_mode, uint32_t omod, bool clamp, bool fp16_ovfl,
                        bool clamp_nan_to_zero) {
  src0 = detail::flush_input_f16(detail::modify_f16(src0, abs0, neg0), denorm_mode);
  src1 = detail::flush_input_f16(detail::modify_f16(src1, abs1, neg1), denorm_mode);
  src2 = detail::flush_input_f16(detail::modify_f16(src2, abs2, neg2), denorm_mode);

  const double multiplicand = static_cast<double>(util::f16_to_f32(src0));
  const double multiplier = static_cast<double>(util::f16_to_f32(src1));
  const double addend = static_cast<double>(util::f16_to_f32(src2));
  double value = std::fma(multiplicand, multiplier, addend);
  if (value == 0.0) {
    // F16 products cannot underflow in double: zero here is exact. Cancellation
    // uses the guest rounding mode; matching zero signs retain their sign.
    const bool negative_product = ((src0 ^ src1) & 0x8000u) != 0;
    const bool matching_zeros =
        (src2 & 0x7fffu) == 0 && negative_product == ((src2 & 0x8000u) != 0);
    const bool negative_zero = matching_zeros ? negative_product : round_mode == 2;
    value = negative_zero ? -0.0 : 0.0;
  }
  uint16_t result =
      pseudo_scalar::round_f16_result(value, round_mode, omod, clamp, fp16_ovfl, clamp_nan_to_zero);
  if ((denorm_mode & 2u) == 0 && (result & 0x7c00u) == 0 && (result & 0x03ffu) != 0)
    result &= 0x8000u;
  return finalize_omod_f16(result, omod);
}

/// @brief Execute an F64 fused multiply-add under MODE.FP_ROUND and MODE.FP_DENORM.
inline uint64_t fma_f64(uint64_t src0, uint64_t src1, uint64_t src2, uint32_t round_mode,
                        uint32_t denorm_mode) {
  if ((denorm_mode & 1u) == 0) {
    src0 = detail::flush_f64(src0);
    src1 = detail::flush_f64(src1);
    src2 = detail::flush_f64(src2);
  }

  uint64_t result;
  {
    detail::ScopedFenv environment(round_mode);
    const double value = std::fma(std::bit_cast<double>(src0), std::bit_cast<double>(src1),
                                  std::bit_cast<double>(src2));
    result = std::bit_cast<uint64_t>(value);
  }
  if ((denorm_mode & 2u) == 0)
    result = detail::flush_f64(result);
  return result;
}

/// @brief Apply F64 OMOD/CLAMP under the architectural rounding mode.
/// @details A nonzero OMOD flushes a denormal result and converts either signed zero to +0.
inline uint64_t finish_f64(uint64_t value, uint32_t round_mode, uint32_t omod, bool clamp,
                           bool clamp_nan_to_zero) {
  double result;
  {
    detail::ScopedFenv environment(round_mode);
    result = std::bit_cast<double>(value);
    if (omod == 1)
      result *= 2.0;
    else if (omod == 2)
      result *= 4.0;
    else if (omod == 3)
      result *= 0.5;
    if (clamp) {
      if ((clamp_nan_to_zero && std::isnan(result)) || result <= 0.0)
        result = 0.0;
      else if (result > 1.0)
        result = 1.0;
    }
  }
  return std::bit_cast<uint64_t>(finalize_omod_f64(result, omod));
}

/// @brief Scale an exact unsigned 53-bit significand using round-toward-zero.
/// @details The host floating-point environment is restored before returning.
inline uint64_t scale_u53_f64_rtz(uint64_t significand, int exponent) {
  detail::ScopedFenv environment(3);
  return std::bit_cast<uint64_t>(std::ldexp(static_cast<double>(significand), exponent));
}

} // namespace rocjitsu::amdgpu::fp_mode

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// MODE-aware arithmetic for memory atomics and gfx10 interpolation.
// Host FP controls remain unchanged; guest rounding is explicit.
// Migrated VALU arithmetic is implemented by rj-arith.
#pragma once

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/shared/denormal.h"
#include "rocjitsu/isa/arch/amdgpu/shared/denormal_value.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_rounding.h"
#include "rocjitsu/isa/arch/amdgpu/shared/input_denormal.h"
#include "rocjitsu/isa/arch/amdgpu/shared/output_modifier.h"
#include "rocjitsu/isa/arch/amdgpu/shared/pseudo_scalar.h"
#include "rocjitsu/isa/isa_family.h"
#include "util/data_types.h"
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <type_traits>

namespace rocjitsu::amdgpu::fp_mode {
enum class ScalarAtomicOp { FADD, FMIN, FMAX, FCMPSWAP };

/// @brief Execute scalar floating atomics with explicit ISA policy and bit-preserving selection.
/// @details FCMPSWAP flushes only comparison inputs and selects the original source or memory
/// bits. RDNA3/3.5 ISA 13.2 and CDNA2 ISA 9.5.2 / CDNA3/4 ISA 9.2.2 instead specify flushing
/// the CompareStore result. Captures on gfx1010/gfx1030/gfx1100 LDS (F32/F64) and gfx1100
/// global F32 contradict that rule on those paths, as do MI350X LDS F32/F64. CDNA1-3 and
/// RDNA3.5 LDS F32/F64, RDNA1/2 cache F32/F64, RDNA3.5 cache F32, and gfx1100 FLAT/BUFFER F32
/// inherit this policy without equivalent captures. CDNA5/RDNA4 expose no FCMPSWAP instructions
/// in this tree. Every RDNA capture used FP_DENORM 0 or 3, endpoints that cannot separate
/// input-controlled from output-controlled flushing. Denormal memory compared against +0 with a
/// normal replacement would, under FP_DENORM 1 or 2: this policy swaps only in mode 2.
template <typename Bits>
Bits atomic_scalar(ScalarAtomicOp operation, Bits old_bits, Bits source_bits, Bits compare_bits,
                   uint32_t denorm_mode, bool legacy_minmax, bool source_nan_first = false,
                   bool minmax_snan_wins = false) {
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
    // classify NaNs explicitly so equal payloads never cause a swap. Denormal
    // flushing applies only to the comparison; selection preserves the original
    // source bits on a match and original memory bits on a mismatch.
    const bool equal =
        old_input == compare_input || ((old_input & ~kSign) == 0 && (compare_input & ~kSign) == 0);
    return !is_nan(old_input) && !is_nan(compare_input) && equal ? source_bits : old_bits;
  }
  if (operation == ScalarAtomicOp::FADD) {
    // DS and cache ADD use fixed RNE, independent of both MODE.round and the host.
    // Select NaNs explicitly so host operand scheduling cannot change payload priority.
    if (source_nan_first && is_nan(source_input)) {
      return source_input | kQuiet;
    }
    if (is_nan(old_input)) {
      return old_input | kQuiet;
    }
    if (is_nan(source_input)) {
      return source_input | kQuiet;
    }
    if ((old_input & ~kSign) == kExponent && (source_input & ~kSign) == kExponent &&
        ((old_input ^ source_input) & kSign)) {
      return kSign | kExponent | kQuiet;
    }
    volatile Float old_value = std::bit_cast<Float>(old_input);
    volatile Float source_value = std::bit_cast<Float>(source_input);
    const Bits result = std::bit_cast<Bits>(static_cast<Float>(old_value + source_value));
    return (denorm_mode & 2u) ? result : flush(result);
  }

  assert((operation == ScalarAtomicOp::FMIN || operation == ScalarAtomicOp::FMAX) &&
         "unsupported scalar floating-point atomic operation");

  // legacy_minmax is the emulator's current CDNA1-2 / RDNA1-3.5 policy: return the original
  // selected bits and propagate SNaNs. MI300A, MI350X and gfx1100 DS hardware instead return
  // the flushed winner and a quieted SNaN, the data operand's first; CDNA5 / RDNA4 flush too.
  const Bits old_result = legacy_minmax ? old_bits : old_input;
  const Bits source_result = legacy_minmax ? source_bits : source_input;
  auto is_snan = [&](Bits bits) { return is_nan(bits) && !(bits & kQuiet); };
  if (legacy_minmax) {
    if (is_snan(old_input)) {
      return old_input | kQuiet;
    }
    if (is_snan(source_input)) {
      return source_input | kQuiet;
    }
  } else if (minmax_snan_wins) {
    if (is_snan(source_input)) {
      return source_input | kQuiet;
    }
    if (is_snan(old_input)) {
      return old_input | kQuiet;
    }
  }
  if (is_nan(old_input)) {
    return is_nan(source_input) ? (legacy_minmax ? old_input : source_input) | kQuiet
                                : source_result;
  }
  if (is_nan(source_input)) {
    return old_result;
  }

  // Compare IEEE bit patterns directly. This also avoids host DAZ affecting a
  // preserved denormal comparison and orders -0 before +0 without host fmin/fmax.
  const Bits old_order = (old_input & kSign) ? ~old_input : old_input ^ kSign;
  const Bits source_order = (source_input & kSign) ? ~source_input : source_input ^ kSign;
  const bool select_source =
      operation == ScalarAtomicOp::FMIN ? source_order < old_order : source_order > old_order;
  return select_source ? source_result : old_result;
}

namespace detail {

// Fix the element format before processing either packed component.
template <bool Bf16>
[[gnu::always_inline]] inline uint32_t add_packed_16(uint32_t old_val, uint32_t src_val,
                                                     uint32_t denorm_mode, bool source_nan_first) {
  constexpr uint16_t exponent_mask = Bf16 ? 0x7f80u : 0x7c00u;
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
    constexpr uint16_t quiet_bit = Bf16 ? 0x0040u : 0x0200u;
    auto is_nan = [&](uint16_t bits) { return (bits & 0x7fffu) > exponent_mask; };
    uint16_t sum;
    // Float memory add propagates the first NaN, quieting it without changing
    // the sign or payload. Invalid infinity addition produces negative QNaN.
    if (source_nan_first && is_nan(b)) {
      sum = b | quiet_bit;
    } else if (is_nan(a)) {
      sum = a | quiet_bit;
    } else if (is_nan(b)) {
      sum = b | quiet_bit;
    } else if ((a & 0x7fffu) == exponent_mask && (b & 0x7fffu) == exponent_mask &&
               (a ^ b) == 0x8000u) {
      sum = 0x8000u | exponent_mask | quiet_bit;
    } else {
      if constexpr (Bf16) {
        // Sums near a BF16 midpoint fit exactly in FP32, so this cannot
        // double-round. Default host rounding also supplies the zero sign.
        sum = util::f32_to_bf16_rne(util::bf16_to_f32(a) + util::bf16_to_f32(b));
      } else {
#if defined(__FLT16_MANT_DIG__)
        // NaNs were selected above. Native half conversions let the compiler
        // use F16C where available, preserving subnormals under default controls.
        const float left = static_cast<float>(std::bit_cast<_Float16>(a));
        const float right = static_cast<float>(std::bit_cast<_Float16>(b));
        sum = std::bit_cast<uint16_t>(static_cast<_Float16>(left + right));
#else
        sum = util::f32_to_f16(util::f16_to_f32(a) + util::f16_to_f32(b));
#endif
      }
    }
    if (!(denorm_mode & 2u)) {
      sum = flush_denorm(sum);
    }
    result |= uint32_t{sum} << shift;
  }
  return result;
}

} // namespace detail

/// @brief Add packed F16/BF16 components using float-memory-atomic policy.
/// @details CDNA5 ISA chapter 12 fixes rounding to nearest-even and specifies
/// NaN selection. FP16_OVFL applies to VALU results, not memory atomics. The
/// caller supplies DS input/output denormal controls, or 3 for no flushing.
[[gnu::always_inline]] inline uint32_t atomic_add_packed_16(uint32_t old_val, uint32_t src_val,
                                                            bool bf16, uint32_t denorm_mode,
                                                            bool source_nan_first = false) {
  return bf16 ? detail::add_packed_16<true>(old_val, src_val, denorm_mode, source_nan_first)
              : detail::add_packed_16<false>(old_val, src_val, denorm_mode, source_nan_first);
}

// gfx10 interpolation still uses these MODE-aware arithmetic stages.

namespace detail {
/// @brief Flush a raw F16 source under MODE input-denormal control.
inline uint16_t flush_input_f16(uint16_t value, uint32_t denorm_mode) {
  return static_cast<uint16_t>(input_denormal::flush_input<fp_format::F16>(
      uint32_t{value}, input_denormal::Policy::make(denorm_mode)));
}
} // namespace detail

/// @brief Whether floating operations with exception support quiet signaling NaNs.
inline bool quiets_nan(rocjitsu::IsaFamily arch, bool ieee_mode) {
  return arch == rocjitsu::IsaFamily::Rdna4 || arch == rocjitsu::IsaFamily::Cdna5 || ieee_mode;
}

/// @brief Return the OMOD value supported by an ordinary floating-point result.
inline uint32_t effective_omod(rocjitsu::IsaFamily arch, uint32_t denorm_mode, bool ieee_mode,
                               uint32_t omod) {
  if (omod == 0) {
    return 0;
  }
  if (arch == rocjitsu::IsaFamily::Rdna4 || arch == rocjitsu::IsaFamily::Cdna5) {
    return omod;
  }
  return (denorm_mode & 2u) == 0 && !ieee_mode ? omod : 0;
}

namespace detail {
/// Round an F32-source FMA directly to F16 in a clean nearest environment.
inline uint16_t fma_f32_to_f16_nearest_environment(float a, float b, float c, uint32_t round_mode,
                                                   bool clamp, bool fp16_ovfl,
                                                   bool clamp_nan_to_zero) {
  double value;
  if (std::isfinite(a) && std::isfinite(b) && std::isfinite(c)) {
    const auto exact = add_exact(double(a) * double(b), double(c));
    value = detail::round_to_odd(exact);
    if (exact.value == 0 && exact.error == 0) {
      // Finite F32 products are exact, normal F64 values or signed zero.
      // Matching signed zeros keep their sign; cancellation and opposite
      // signed zeros are negative only when rounding toward negative infinity.
      const uint32_t product_sign = (std::bit_cast<uint32_t>(a) ^ std::bit_cast<uint32_t>(b)) >> 31;
      const uint32_t addend_bits = std::bit_cast<uint32_t>(c);
      const bool matching_zeros =
          (addend_bits & 0x7fffffffu) == 0 && product_sign == (addend_bits >> 31);
      const bool negative = matching_zeros ? product_sign != 0 : (round_mode & 3u) == 2u;
      value = std::bit_cast<double>(uint64_t{negative} << 63);
    }
  } else {
    const uint32_t a_abs = std::bit_cast<uint32_t>(a) & 0x7fffffffu;
    const uint32_t b_abs = std::bit_cast<uint32_t>(b) & 0x7fffffffu;
    // The invalid product wins even when the addend supplies a NaN payload.
    const bool invalid_product =
        (a_abs == 0 && b_abs == 0x7f800000u) || (b_abs == 0 && a_abs == 0x7f800000u);
    value = invalid_product ? std::bit_cast<double>(uint64_t{0xfff8000000000000})
                            : std::fma(double(a), double(b), double(c));
  }
  return pseudo_scalar::round_f16_result(value, round_mode, clamp, fp16_ovfl, clamp_nan_to_zero);
}
} // namespace detail

// Apply interpolation's input flushing, NaN policy and guest rounding without
// changing host controls. Finite arithmetic uses FP64 with a retained residual.
inline float fma_f32(float a, float b, float c, uint32_t round_mode, uint32_t denorm_mode,
                     rocjitsu::IsaFamily arch, bool ieee_mode) {
  auto input_bits = [denorm_mode](float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    return !(denorm_mode & 1u) && (bits & 0x7f800000u) == 0 ? bits & 0x80000000u : bits;
  };
  const uint32_t a_bits = input_bits(a), b_bits = input_bits(b), c_bits = input_bits(c);
  const uint32_t a_abs = a_bits & 0x7fffffffu, b_abs = b_bits & 0x7fffffffu;
  if ((a_abs == 0 && b_abs == 0x7f800000u) || (b_abs == 0 && a_abs == 0x7f800000u)) {
    return std::bit_cast<float>(0xffc00000u);
  }
  for (const uint32_t bits : {a_bits, b_bits, c_bits}) {
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
      return std::bit_cast<float>(bits | (quiets_nan(arch, ieee_mode) ? 0x00400000u : 0u));
    }
  }
  a = std::bit_cast<float>(a_bits);
  b = std::bit_cast<float>(b_bits);
  c = std::bit_cast<float>(c_bits);
  if (std::isfinite(a) && std::isfinite(b) && std::isfinite(c)) {
    return detail::fma_finite_f32(a, b, c, round_mode & 3u, !(denorm_mode & 2u));
  }
  const float result = std::fma(a, b, c);
  return std::isnan(result) ? std::bit_cast<float>(0xffc00000u) : result;
}

/// @brief Scale an F32 result by OMOD through the shared raw-bit stage.
/// @details Zeros and tiny results become +0 before scaling; a normal result that
/// becomes tiny when halved keeps its sign as a zero. Overflow rounds by
/// `round_mode` (MODE.FP_ROUND); TRANS callers pass nearest-even.
inline float apply_omod_f32(float value, uint32_t omod, uint32_t round_mode) {
  return std::bit_cast<float>(
      output_modifier::scale_f32(std::bit_cast<uint32_t>(value), omod, round_mode));
}

} // namespace rocjitsu::amdgpu::fp_mode

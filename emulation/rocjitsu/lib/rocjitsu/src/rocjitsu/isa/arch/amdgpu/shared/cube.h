// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file cube.h
/// @brief Shared scalar and SIMD cube coordinate instruction semantics.

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "util/simd.h"

#include <bit>
#include <cstdint>
#include <type_traits>

namespace rocjitsu::amdgpu::cube {

/// @brief Cube face, S/T coordinate, and signed major-axis operations.
enum class Operation : uint8_t { ID, SC, TC, MA };

namespace detail {

inline constexpr uint32_t kSign = 0x80000000u;
inline constexpr uint32_t kMagnitude = 0x7fffffffu;
inline constexpr uint32_t kInfinity = 0x7f800000u;
inline constexpr uint32_t kExponentStep = 0x00800000u;
inline constexpr uint32_t kMaxFinite = 0x7f7fffffu;
inline constexpr uint32_t kDoubleOverflow = 0x7f000000u;
inline constexpr uint32_t kQuietNaN = 0x00400000u;
inline constexpr uint32_t kFaceNegativeZ = 0x40a00000u;
inline constexpr uint32_t kFacePositiveZ = 0x40800000u;
inline constexpr uint32_t kFaceNegativeY = 0x40400000u;
inline constexpr uint32_t kFacePositiveY = 0x40000000u;
inline constexpr uint32_t kFaceNegativeX = 0x3f800000u;
inline constexpr uint32_t kFacePositiveX = 0x00000000u;

template <typename Bits, typename Mask> Bits select(Mask condition, Bits yes, Bits no) {
  if constexpr (std::is_same_v<Bits, uint32_t>) {
    return condition ? yes : no;
  } else {
    util::stdx::where(condition, no) = yes;
    return no;
  }
}

// Comparisons flush subnormal magnitudes regardless of MODE, but SC/TC copy
// the selected raw source. Integer comparisons also preserve signaling NaNs.
template <typename Bits> Bits comparison_magnitude(Bits bits) {
  const Bits magnitude = bits & Bits(kMagnitude);
  return select<Bits>(magnitude < Bits(kExponentStep), Bits(0), magnitude);
}

template <typename Bits> Bits overflow(Bits sign, uint32_t round_mode) {
  Bits magnitude = Bits(kInfinity);
  if (round_mode == 1)
    magnitude = select<Bits>(sign != Bits(0), Bits(kMaxFinite), magnitude);
  else if (round_mode == 2)
    magnitude = select<Bits>(sign == Bits(0), Bits(kMaxFinite), magnitude);
  else if (round_mode == 3)
    magnitude = Bits(kMaxFinite);
  return sign | magnitude;
}

template <Operation Op, typename Bits>
Bits execute_bits(Bits x, Bits y, Bits z, uint32_t round_mode, bool quiet_nan) {
  const Bits ax = comparison_magnitude(x);
  const Bits ay = comparison_magnitude(y);
  const Bits az = comparison_magnitude(z);
  const auto z_axis = (az <= Bits(kInfinity)) && (ax <= Bits(kInfinity)) &&
                      (ay <= Bits(kInfinity)) && (az >= ax) && (az >= ay);
  const auto y_axis = (ay <= Bits(kInfinity)) && (ax <= Bits(kInfinity)) && (ay >= ax);
  const Bits major = select<Bits>(z_axis, z, select<Bits>(y_axis, y, x));
  const Bits magnitude = comparison_magnitude(major);
  const auto negative = ((major & Bits(kSign)) != Bits(0)) && (magnitude != Bits(0)) &&
                        (magnitude <= Bits(kInfinity));
  Bits result;
  if constexpr (Op == Operation::ID) {
    // Z wins ties, then Y, then X. A zero or unordered major is nonnegative.
    result = select<Bits>(
        z_axis, select<Bits>(negative, Bits(kFaceNegativeZ), Bits(kFacePositiveZ)),
        select<Bits>(y_axis, select<Bits>(negative, Bits(kFaceNegativeY), Bits(kFacePositiveY)),
                     select<Bits>(negative, Bits(kFaceNegativeX), Bits(kFacePositiveX))));
  } else if constexpr (Op == Operation::SC) {
    const Bits sign = select<Bits>(negative, Bits(kSign), Bits(0));
    result = select<Bits>(z_axis, x ^ sign, select<Bits>(y_axis, x, z ^ sign ^ Bits(kSign)));
  } else if constexpr (Op == Operation::TC) {
    const Bits sign = select<Bits>(negative, Bits(kSign), Bits(0));
    result = select<Bits>(!z_axis && y_axis, z ^ sign, y ^ Bits(kSign));
  } else {
    const Bits sign = major & Bits(kSign);
    // Multiplication by two is exact until overflow. Directed rounding chooses
    // the appropriate finite bound; only round-down retains negative zero.
    result = select<Bits>(magnitude >= Bits(kDoubleOverflow), overflow(sign, round_mode),
                          major + Bits(kExponentStep));
    result = select<Bits>(magnitude >= Bits(kInfinity), major, result);
    result = select<Bits>(magnitude == Bits(0), round_mode == 2 ? sign : Bits(0), result);
  }
  if (quiet_nan)
    result = select<Bits>((result & Bits(kMagnitude)) > Bits(kInfinity), result | Bits(kQuietNaN),
                          result);
  return result;
}

} // namespace detail

/// @brief Execute a cube operation after source modifiers and before output modifiers.
/// @details Scalar and SIMD paths share bit-level selection, rounding and NaN handling.
template <Operation Op, typename Float>
Float execute(Float x, Float y, Float z, uint32_t round_mode, rj_code_arch_t arch, bool ieee_mode) {
  using Bits = std::conditional_t<std::is_same_v<Float, float>, uint32_t, util::native<uint32_t>>;
  const bool quiet_nan =
      arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5 || ieee_mode;
  return std::bit_cast<Float>(
      detail::execute_bits<Op>(std::bit_cast<Bits>(x), std::bit_cast<Bits>(y),
                               std::bit_cast<Bits>(z), round_mode, quiet_nan));
}

/// @brief Apply cube OMOD independently of the denormal MODE bits.
/// @details Flushes subnormals before scaling, preserves NaN payloads, and uses
/// guest rounding on overflow.
template <typename Float>
Float apply_omod(Float value, uint32_t round_mode, rj_code_arch_t arch, bool ieee_mode,
                 uint32_t omod) {
  omod = fp_mode::effective_omod(arch, 0, ieee_mode, omod);
  if (omod == 0)
    return value;
  using Bits = std::conditional_t<std::is_same_v<Float, float>, uint32_t, util::native<uint32_t>>;
  const Bits bits = std::bit_cast<Bits>(value);
  const Bits sign = bits & Bits(detail::kSign);
  const Bits exponent = bits & Bits(detail::kInfinity);
  Bits result;
  if (omod == 3) {
    result = detail::select<Bits>(exponent == Bits(detail::kExponentStep), sign,
                                  bits - Bits(detail::kExponentStep));
  } else {
    result = detail::select<Bits>(
        exponent >= Bits(detail::kInfinity - omod * detail::kExponentStep),
        detail::overflow(sign, round_mode), bits + Bits(omod * detail::kExponentStep));
  }
  result = detail::select<Bits>(exponent == Bits(detail::kInfinity), bits, result);
  result = detail::select<Bits>(exponent == Bits(0), Bits(0), result);
  return std::bit_cast<Float>(result);
}

} // namespace rocjitsu::amdgpu::cube

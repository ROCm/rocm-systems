// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP32/FP64 encoding traits used by selection algorithms.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "common/bits.h"
#include "fp/float_output.h"
#include <stdint.h>
#include <type_traits>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Raw IEEE binary32/binary64 storage and classification masks.
template <typename Float> class FloatBits {
public:
  static_assert(std::is_same_v<Float, float> || std::is_same_v<Float, double>);
  using Format = IeeeFormat<sizeof(Float) * 8>;
  using UInt = typename Format::Bits;
  static constexpr unsigned fraction_bits = Format::fraction;
  static constexpr UInt sign = Format::sign;
  static constexpr UInt magnitude = sign - 1;
  static constexpr UInt infinity = Format::infinity;
  static constexpr UInt quiet = Format::quiet;

  // Classify either signaling or quiet NaN from its raw encoding.
  static bool is_nan(UInt bits) { return (bits & magnitude) > infinity; }

  // Copy a floating-point representation into its matching unsigned integer type.
  static UInt bits(Float value) { return rocjitsu::arith::bit_cast<UInt>(value); }

  // Copy an unsigned bit pattern into its matching floating-point type.
  static Float value(UInt bits) { return rocjitsu::arith::bit_cast<Float>(bits); }
};

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE

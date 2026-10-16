// SPDX-License-Identifier: MIT

// Conversions between FP32 and signed/unsigned 32-bit integers.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include <cmath>
#include <stdint.h>

namespace rocjitsu::arith {

enum class Conversion32 {
  SignedToFloat,
  UnsignedToFloat,
  FloatToSigned,
  FloatToUnsigned,
  Nearest,
  Floor
};

// Truncate and saturate to 32-bit integer bits without out-of-range casts.
// Signed NaNs yield nan_result; unsigned NaNs and negative values yield zero.
template <bool Signed> uint32_t truncate_integer(float x, uint32_t nan_result = 0) {
  // Check NaN and range before any C++ float-to-integer cast, whose out-of-range behavior is
  // undefined.
  if constexpr (Signed) {
    return std::isnan(x)         ? nan_result
           : x >= 2147483648.0f  ? uint32_t(INT32_MAX)
           : x <= -2147483648.0f ? uint32_t(INT32_MIN)
                                 : uint32_t(int32_t(x));
  } else {
    return !(x > 0) ? 0 : x >= 4294967296.0f ? UINT32_MAX : uint32_t(x);
  }
}

// Convert FP32 and 32-bit integers. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Conversion32 Op>
void conversion32_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a);

// Convert FP32 and 32-bit integers. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Conversion32 Op>
void conversion32_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a);

} // namespace rocjitsu::arith

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Conversions between FP16 and FP32 or 16-bit integers.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

enum class Conversion16 {
  SignedToHalf,
  UnsignedToHalf,
  HalfToSigned,
  HalfToUnsigned,
  FloatToHalf,
  HalfToFloat
};

// Convert FP16, FP32, or 16-bit integers. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Conversion16 Op>
void conversion16_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                            const uint32_t *a, bool quiet_nan);

// Convert FP16, FP32, or 16-bit integers. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Conversion16 Op>
void conversion16_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                            const uint32_t *a, bool quiet_nan);

} // namespace rocjitsu::arith

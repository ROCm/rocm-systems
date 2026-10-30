// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Conversions between FP32 and signed/unsigned 32-bit integers.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

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

// Convert FP32 and 32-bit integers. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Conversion32 Op>
void conversion32_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                            const uint32_t *a);

// Convert FP32 and 32-bit integers. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Conversion32 Op>
void conversion32_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                            const uint32_t *a);

} // namespace rocjitsu::arith

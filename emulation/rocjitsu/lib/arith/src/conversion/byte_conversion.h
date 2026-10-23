// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Byte-to-FP32 conversion, signed nibble offsets, and FP32-to-byte packing.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Convert selected bytes to FP32. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Byte>
void byte_conversion_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a);

// Convert selected bytes to FP32. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Byte>
void byte_conversion_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a);

// Convert selected bytes to FP32. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void nibble_offset_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a);

// Convert selected bytes to FP32. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void nibble_offset_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a);

// Convert selected bytes to FP32. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void byte_pack_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                         const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Convert selected bytes to FP32. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void byte_pack_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                         const uint32_t *a, const uint32_t *b, const uint32_t *c);

} // namespace rocjitsu::arith

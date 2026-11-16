// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// SIMD backend declarations for arithmetic right shift and saturating byte packing.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Replace the selected destination half for active lanes; controls must be validated.
template <bool Signed>
void byte_shift_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                          const uint32_t *b, const uint32_t *c);

// Replace the selected destination half for active lanes; controls must be validated.
template <bool Signed>
void byte_shift_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                          const uint32_t *b, const uint32_t *c);

} // namespace rocjitsu::arith

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// SIMD backend declarations for floating-point class comparisons.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Test floating-point classes.
// Requires the named CPU features and validated instruction controls.
template <int Bits>
uint32_t class_x86_64_v3(uint32_t mode, const uint32_t *const *a, const uint32_t *b);

// Test floating-point classes.
// Requires the named CPU features and validated instruction controls.
template <int Bits>
uint32_t class_x86_64_v4(uint32_t mode, const uint32_t *const *a, const uint32_t *b);

} // namespace rocjitsu::arith

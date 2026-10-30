// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Cross-ISA backend declarations for division fixup.
#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Apply numeric FIXUP with nearest guest rounding; FP status is computed separately.
// Preserve inactive lanes and support destination/input aliasing.
// Requires the named CPU features and validated instruction controls.
template <int Width, bool Rdna3Half = false>
void fixup_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint64_t flags, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c);

// Apply numeric FIXUP with nearest guest rounding; FP status is computed separately.
// Preserve inactive lanes and support destination/input aliasing.
// Requires the named CPU features and validated instruction controls.
template <int Width, bool Rdna3Half = false>
void fixup_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint64_t flags, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c);

// Wave contribution from the same operand layout as numeric FIXUP. CLAMP
// suppresses causes; otherwise only EXEC-selected lanes contribute.
template <int Width>
uint32_t fixup_exceptions_x86_64_v3(uint32_t exec_mask, uint32_t mode, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);

// Return the OR of FP causes from active lanes; CLAMP suppresses all causes.
// Uses the numeric FIXUP operand layout without writing any operands.
// Requires x86-64-v4 and validated instruction controls.
template <int Width>
uint32_t fixup_exceptions_x86_64_v4(uint32_t exec_mask, uint32_t mode, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c);

} // namespace rocjitsu::arith

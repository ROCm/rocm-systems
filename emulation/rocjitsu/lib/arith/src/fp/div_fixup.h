// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Cross-ISA backend declarations for division fixup.
#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Apply numeric FIXUP with guest rounding; FP status is computed separately.
// Preserve inactive lanes and support destination/input aliasing. CDNA selects
// promoted FP16 scaling and nearest narrowing instead of the RDNA output stage.
// Requires the named CPU features and validated instruction controls.
template <int Width, bool Rdna3 = false>
void fixup_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint64_t flags, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                     bool cdna = false);

// Apply numeric FIXUP with guest rounding; FP status is computed separately.
// Preserve inactive lanes and support destination/input aliasing. CDNA selects
// promoted FP16 scaling and nearest narrowing instead of the RDNA output stage.
// Requires the named CPU features and validated instruction controls.
template <int Width, bool Rdna3 = false>
void fixup_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint64_t flags, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
                     bool cdna = false);

// Wave contribution from the numeric operand layout. Legacy mode retains
// pre-CLAMP causes; the default policy suppresses them under CLAMP. Guest
// input flushing applies before classification; only EXEC lanes contribute.
template <int Width>
uint32_t fixup_exceptions_x86_64_v3(uint32_t exec_mask, uint32_t mode, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c,
                                    uint64_t flags = 0, bool legacy = false);

// Return active-lane FP causes with the same legacy/flush policy as the AVX2 backend.
// Uses the numeric FIXUP operand layout without writing any operands.
// Requires x86-64-v4 and validated instruction controls.
template <int Width>
uint32_t fixup_exceptions_x86_64_v4(uint32_t exec_mask, uint32_t mode, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c,
                                    uint64_t flags = 0, bool legacy = false);

} // namespace rocjitsu::arith

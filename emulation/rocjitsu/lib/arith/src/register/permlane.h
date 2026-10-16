// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Lane permutations across rows, half-waves, and whole waves.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {
// Permute source lanes. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Cross, bool Var>
void permlane_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                        const uint32_t *b, uint32_t lo, uint32_t hi);

// Permute source lanes. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Cross, bool Var>
void permlane_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                        const uint32_t *b, uint32_t lo, uint32_t hi);

} // namespace rocjitsu::arith

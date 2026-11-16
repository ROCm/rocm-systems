// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Lane permutations across rows, half-waves, and whole waves.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {
enum class GroupPermutation { Bcast, Up, Down, Xor };

// Route within normalized power-of-two groups; stage reads before masked writes.
void group_permutation_x86_64_v3(GroupPermutation op, int lanes, uint64_t exec_mask, uint32_t *d,
                                 const uint32_t *a, uint32_t selector, uint32_t width);
void group_permutation_x86_64_v4(GroupPermutation op, int lanes, uint64_t exec_mask, uint32_t *d,
                                 const uint32_t *a, uint32_t selector, uint32_t width);

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

// Evaluate permlane64 for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void permlane64_x86_64_v3(uint64_t exec_mask, uint32_t *d, const uint32_t *a);

// XOR each selected lane with the scalar selector, preserving inactive destinations.
void permlane_index_x86_64_v3(int lanes, uint64_t exec_mask, uint32_t *d, const uint32_t *a,
                              uint32_t group_width);
// XOR each selected lane with the scalar group_width, preserving inactive destinations.
void permlane_index_x86_64_v4(int lanes, uint64_t exec_mask, uint32_t *d, const uint32_t *a,
                              uint32_t group_width);
} // namespace rocjitsu::arith

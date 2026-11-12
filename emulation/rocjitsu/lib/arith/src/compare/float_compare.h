// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Cross-ISA backend declarations for float compare.
#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate ordered/unordered FP predicates.
// Requires the named CPU features and validated instruction controls.
template <int Bits, unsigned Predicate>
uint32_t float_compare_x86_64_v3(uint32_t mode, bool flush, const uint32_t *const *a,
                                 const uint32_t *const *b);

// Evaluate ordered/unordered FP predicates.
// Requires the named CPU features and validated instruction controls.
template <int Bits, unsigned Predicate>
uint32_t float_compare_x86_64_v4(uint32_t mode, bool flush, const uint32_t *const *a,
                                 const uint32_t *const *b);

} // namespace rocjitsu::arith

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Cross-ISA backend declarations for integer compare.
#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Compare signed/unsigned integers.
// Requires the named CPU features and validated instruction controls.
template <int Bits, bool Signed, unsigned Predicate>
uint32_t integer_compare_x86_64_v3(uint32_t mode, const uint32_t *const *a,
                                   const uint32_t *const *b);

// Compare signed/unsigned integers.
// Requires the named CPU features and validated instruction controls.
template <int Bits, bool Signed, unsigned Predicate>
uint32_t integer_compare_x86_64_v4(uint32_t mode, const uint32_t *const *a,
                                   const uint32_t *const *b);

} // namespace rocjitsu::arith

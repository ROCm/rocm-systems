// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Cross-ISA backend declarations for float compare.
#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate a full Wave32/Wave64 predicate, selected by the result-mask type.
// Requires the named CPU features and validated instruction controls.
template <int Bits, unsigned Predicate, typename Mask = uint32_t>
Mask float_compare_x86_64_v3(uint32_t mode, bool flush, const uint32_t *const *a,
                             const uint32_t *const *b);

// Evaluate a full Wave32/Wave64 predicate, selected by the result-mask type.
// Requires the named CPU features and validated instruction controls.
template <int Bits, unsigned Predicate, typename Mask = uint32_t>
Mask float_compare_x86_64_v4(uint32_t mode, bool flush, const uint32_t *const *a,
                             const uint32_t *const *b);

} // namespace rocjitsu::arith

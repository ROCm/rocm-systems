// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Wordwise boolean operations, including three-input truth tables.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

enum class Boolean { And, Or, Xor, Not, Xnor };

// Evaluate wordwise boolean operations. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Boolean Op, bool Half>
void boolean_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                       const uint32_t *b);

// Evaluate wordwise boolean operations. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Boolean Op, bool Half>
void boolean_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                       const uint32_t *b);

} // namespace rocjitsu::arith

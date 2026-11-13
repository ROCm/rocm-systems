// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Three-input integer arithmetic, selection, and shift/boolean combinations.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#ifndef RJ_ARITH_INTEGER_TERNARY_H_
#define RJ_ARITH_INTEGER_TERNARY_H_

#include <stdint.h>

namespace rocjitsu::arith {

enum class IntegerTernary { ShiftAdd, AddShift, ShiftOr, AndOr, Or3, Xor3, XorAdd, Lerp };

// Evaluate three-input integer operations. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <IntegerTernary Op>
void integer_ternary_x86_64_v3(uint32_t exec_mask, uint32_t *d, const uint32_t *a,
                               const uint32_t *b, const uint32_t *c);

// Evaluate three-input integer operations. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <IntegerTernary Op>
void integer_ternary_x86_64_v4(uint32_t exec_mask, uint32_t *d, const uint32_t *a,
                               const uint32_t *b, const uint32_t *c);

} // namespace rocjitsu::arith

#endif

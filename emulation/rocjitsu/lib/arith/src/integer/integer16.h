// SPDX-License-Identifier: MIT

// 16-bit integer add/subtract, multiply, shifts, and min/max.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

enum class Integer16 { Add, Sub, Min, Max, Mul, ShiftLeft, ShiftRight };

// Evaluate 16-bit integer arithmetic. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Integer16 Op, bool Signed, bool Packed = true>
void integer16_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                         const uint32_t *b);

} // namespace rocjitsu::arith

// SPDX-License-Identifier: MIT

// 32/64-bit logical and arithmetic shifts, including reversed operand forms.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#ifndef RJ_ARITH_SHIFT_H_
#define RJ_ARITH_SHIFT_H_

#include <stdint.h>

namespace rocjitsu::arith {

enum class Shift { Left, LogicalRight, ArithmeticRight };

// Shift integer words. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Bits, Shift Op>
void shift_x86_64_v3(uint32_t exec_mask, uint32_t *const *d, const uint32_t *a,
                     const uint32_t *const *b);

} // namespace rocjitsu::arith

#endif

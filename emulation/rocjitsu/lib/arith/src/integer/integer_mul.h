// SPDX-License-Identifier: MIT

// Low/high-half integer multiplication with signed or unsigned operands.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Multiply integers and select result halves. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Bits, bool Signed, bool High>
void integer_mul_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                           const uint32_t *b);

// Multiply integers and select result halves. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Bits, bool Signed, bool High>
void integer_mul_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                           const uint32_t *b);

} // namespace rocjitsu::arith

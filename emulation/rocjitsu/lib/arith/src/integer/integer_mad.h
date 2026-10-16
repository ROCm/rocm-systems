// SPDX-License-Identifier: MIT

// 24-bit and packed 16-bit integer multiply-add.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Multiply-add narrow integer factors. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <int Bits, bool Signed>
void integer_mad_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                           const uint32_t *b, const uint32_t *c);

} // namespace rocjitsu::arith

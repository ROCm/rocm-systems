// SPDX-License-Identifier: MIT

// Lit-multiply arithmetic and special-value fixup controlled by the third operand.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Multiply with lit special-value fixup. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Rdna3>
void mullit_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint64_t flags, uint32_t *d,
                      const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Multiply with lit special-value fixup. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Rdna3>
void mullit_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint64_t flags, uint32_t *d,
                      const uint32_t *a, const uint32_t *b, const uint32_t *c);

} // namespace rocjitsu::arith

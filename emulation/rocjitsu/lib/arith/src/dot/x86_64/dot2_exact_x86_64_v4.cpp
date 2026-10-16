// SPDX-License-Identifier: MIT

// AVX-512 bit-exact FP16/BF16 DOT2 vector execution and masked result stores.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "dot/x86_64/dot2_exact.h"
#include "dot/x86_64/dot2_exact_x86_64.h"

namespace rocjitsu::arith {

// Evaluate dot2 exact for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void dot2_exact_x86_64_v4(bool bf16, bool gfx11, uint32_t exec_mask, uint32_t modifiers,
                          uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  if (bf16) {
    if (gfx11) {
      exact_dot::dot2<true, true, __m512i>(exec_mask, modifiers, d, a, b, c);
    } else {
      exact_dot::dot2<true, false, __m512i>(exec_mask, modifiers, d, a, b, c);
    }
  } else {
    if (gfx11) {
      exact_dot::dot2<false, true, __m512i>(exec_mask, modifiers, d, a, b, c);
    } else {
      exact_dot::dot2<false, false, __m512i>(exec_mask, modifiers, d, a, b, c);
    }
  }
}

} // namespace rocjitsu::arith

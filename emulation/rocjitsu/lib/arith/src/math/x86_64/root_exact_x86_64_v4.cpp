// SPDX-License-Identifier: MIT

// AVX-512 vector empirical square-root/reciprocal-square-root evaluation and scalar repair.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "math/x86_64/root_exact.h"
#include "math/x86_64/root_exact_x86_64.h"

namespace rocjitsu::arith {
// Evaluate root exact for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
uint32_t root_exact_x86_64_v4(bool reciprocal, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                              uint32_t *d, const uint32_t *a, bool quiet_snan, bool reporting) {
  if (reciprocal) {
    return root_simd::run<true, root_simd::U8>(flags, exec_mask, mode, d, a, quiet_snan, reporting);
  }
  return root_simd::run<false, root_simd::U8>(flags, exec_mask, mode, d, a, quiet_snan, reporting);
}
} // namespace rocjitsu::arith

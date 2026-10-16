// SPDX-License-Identifier: MIT

// x86 SIMD backend declarations for empirical exact FP32 square-root and
// reciprocal-square-root.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {
// Evaluate root exact for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
uint32_t root_exact_x86_64_v3(bool reciprocal, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                              uint32_t *d, const uint32_t *a, bool quiet_snan, bool reporting);

// Evaluate root exact for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
uint32_t root_exact_x86_64_v4(bool reciprocal, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                              uint32_t *d, const uint32_t *a, bool quiet_snan, bool reporting);

} // namespace rocjitsu::arith

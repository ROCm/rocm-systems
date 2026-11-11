// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 vector empirical FP64 reciprocal, reciprocal-square-root and square-root model.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "math/x86_64/trans_f64_exact.h"
#include "math/x86_64/trans_f64_exact_x86_64.h"

namespace rocjitsu::arith {
// Evaluate FP64 RCP, RSQ or SQRT with A ABS/NEG, guest denormal flags, OMOD and CLAMP.
// Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void trans_f64_exact_x86_64_v4(Fp64 op, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                               uint32_t *const *d, const uint32_t *const *a, bool quiet_snan) {
  trans_f64_simd::dispatch<trans_f64_simd::U8, trans_f64_simd::S8>(op, flags, exec_mask, mode, d, a,
                                                                   quiet_snan);
}
} // namespace rocjitsu::arith

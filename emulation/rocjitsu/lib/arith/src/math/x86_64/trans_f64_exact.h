// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86 SIMD backend declarations for the empirical exact FP64 reciprocal,
// reciprocal-square-root and square-root model.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include "fp/fp64.h"

#include <stdint.h>

namespace rocjitsu::arith {
// Evaluate FP64 RCP, RSQ or SQRT with A ABS/NEG, guest denormal flags, OMOD and CLAMP.
// Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void trans_f64_exact_x86_64_v3(Fp64 op, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                               uint32_t *const *d, const uint32_t *const *a, bool quiet_snan);

// Evaluate FP64 RCP, RSQ or SQRT with A ABS/NEG, guest denormal flags, OMOD and CLAMP.
// Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void trans_f64_exact_x86_64_v4(Fp64 op, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                               uint32_t *const *d, const uint32_t *const *a, bool quiet_snan);

} // namespace rocjitsu::arith

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86-64-v3 dispatch for the inherited vector EXP model.
#include "math/x86_64/exp_exact.h"
#include "math/x86_64/exp_exact_x86_64.h"

namespace rocjitsu::arith {

// Evaluate EXP with validated effective controls and preserve inactive output lanes.
void exp_exact_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                         const uint32_t *a, bool quiet_snan) {
  exp_simd::run<exp_simd::U4>(flags, exec_mask, mode, d, a, quiet_snan);
}

} // namespace rocjitsu::arith

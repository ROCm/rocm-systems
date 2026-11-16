// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 vector empirical reciprocal evaluation, corrections, and exceptional-lane repair.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "math/x86_64/rcp_exact.h"
#include "math/x86_64/rcp_exact_x86_64.h"

namespace rocjitsu::arith {
// Evaluate rcp exact for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void rcp_exact_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                         const uint32_t *a, bool quiet_snan) {
  rcp_simd::run<rcp_simd::U4>(flags, exec_mask, mode, d, a, quiet_snan);
}
} // namespace rocjitsu::arith

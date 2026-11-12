// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86 SIMD backend declarations for empirical exact FP32 reciprocal execution.
// Shared vector helpers; entry-point policy and dispatch remain in the family callers.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate rcp exact for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void rcp_exact_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                         const uint32_t *a, bool quiet_snan);

// Evaluate rcp exact for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void rcp_exact_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                         const uint32_t *a, bool quiet_snan);

} // namespace rocjitsu::arith

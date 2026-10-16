// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Packed FP16 FMA/FMAC with independent low/high operand selectors.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate paired FP16 fused multiply-adds. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Accumulate>
void packed_fma_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                          const uint32_t *a, const uint32_t *b, const uint32_t *c);

} // namespace rocjitsu::arith

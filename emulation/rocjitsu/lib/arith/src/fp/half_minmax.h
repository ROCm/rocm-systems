// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16 three-input minimum, maximum, median, and combined selections.
// SIMD backend declaration; instruction policy is shared in minmax.h.

#pragma once

#include "fp/minmax.h"

#include <stdint.h>

namespace rocjitsu::arith {

// Select among three FP16 values. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Minmax3 Op>
void half_minmax3_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                            const uint32_t *a, const uint32_t *b, const uint32_t *c);

} // namespace rocjitsu::arith

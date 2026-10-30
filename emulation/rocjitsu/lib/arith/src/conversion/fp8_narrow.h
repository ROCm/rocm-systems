// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP32 narrowing to FP8/BF8 with nearest or seeded stochastic rounding.
// Internal helpers shared by instruction implementations; not part of the public C API.

// Rounding logic adapted from rocjitsu util/data_types.h, with RDNA4 NaN,
// infinity and subnormal stochastic behavior established by GPU captures.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Narrow FP32 to FP8/BF8. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Bf8, bool Stochastic>
void fp8_narrow_x86_64_v3(uint32_t exec_mask, uint32_t mode, bool saturate, uint32_t *d,
                          const uint32_t *a, const uint32_t *b);

// Narrow FP32 to FP8/BF8. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Bf8, bool Stochastic>
void fp8_narrow_x86_64_v4(uint32_t exec_mask, uint32_t mode, bool saturate, uint32_t *d,
                          const uint32_t *a, const uint32_t *b);

} // namespace rocjitsu::arith

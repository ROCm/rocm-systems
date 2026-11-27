// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 FRACT backend shared by FP32 architecture-policy wrappers.
#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Compute capped fractional parts with validated guest controls and the selected NaN policy.
// Preserve inactive lanes and register overlap, including split-word FP64 aliases.
void fract_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                     const uint32_t *const *a, bool quiet_snan);

} // namespace rocjitsu::arith

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared AVX-512 mantissa dispatch for the existing FP32 numerical models.
#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Normalize finite values to signed [0.5,1), preserving zero/infinity and selected NaN policy.
// Requires validated flags; apply guest controls and preserve masked lanes and register aliases.
void frexp_mant_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                          const uint32_t *const *a, bool quiet_snan);

} // namespace rocjitsu::arith

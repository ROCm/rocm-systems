// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared decoded integer SWMMAC operands and baseline/SIMD backend declarations.
// This header defines the cross-backend data layout, without callable implementations.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

struct SwmmacIntegerInputs {
  int32_t a[16][32];
  int32_t b[64][16];
  int32_t acc[16][16];
  uint8_t selected[16][32];
};

// Accumulate decoded sparse INT8/INT4 operands and write eight 32-lane D
// sections. k is the dense reduction size (32 or 64); clamp saturates each
// stage, otherwise results wrap. Input snapshots must not overlap D.
void swmmac_integer_x86_64_v3(int k, bool clamp, uint32_t *const *d,
                              const SwmmacIntegerInputs &input);
// Accumulate decoded sparse INT8/INT4 operands and write eight 32-lane D
// sections. k is the dense reduction size (32 or 64); clamp saturates each
// stage, otherwise results wrap. Input snapshots must not overlap D.
void swmmac_integer_x86_64_v4(int k, bool clamp, uint32_t *const *d,
                              const SwmmacIntegerInputs &input);

} // namespace rocjitsu::arith

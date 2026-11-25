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

// Raw SIMD backend signatures. Public entry points validate mode and adapt tied D
// into 32-lane sections. All input arrays may overlap the destination registers.
using SwmmacIntegerBackend = void(uint32_t, uint32_t *const *, const uint32_t *const *,
                                  const uint32_t *const *, const uint32_t *const *);

// SIMD backends consume packed architectural operands after public validation.
template <int Bits, int K, int WaveSize> SwmmacIntegerBackend swmmac_integer_x86_64_v3;
template <int Bits, int K, int WaveSize> SwmmacIntegerBackend swmmac_integer_x86_64_v4;

} // namespace rocjitsu::arith

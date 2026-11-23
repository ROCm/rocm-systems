// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared decoded float SWMMAC operands and baseline/SIMD backend declarations.
// This header defines the cross-backend data layout, without callable implementations.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Decoded gfx12 16x16x32 tile for the backends below; each A row holds
// 16 compressed elements, with metadata selecting their dense K coordinates.
struct SwmmacFloatInputs {
  float a[16][16];
  float b[32][16];
  float acc[16][16];
  uint8_t selected[16][16];
};

// Accumulate all decoded products in sparse K order, applying B negation by pair position.
// Updates only input.acc; input arrays must already include any A modifiers.
void swmmac_float_accumulate(uint32_t mode, SwmmacFloatInputs &input);

// Raw SIMD backend signatures. Public entry points validate mode and adapt tied D
// into 32-lane sections. All input arrays may overlap the destination registers.
using Swmmac16Backend = void(uint32_t, uint32_t *const *, const uint32_t *const *,
                             const uint32_t *const *, const uint32_t *const *, bool);
using Swmmac8Backend = void(uint32_t, uint32_t *const *, const uint32_t *const *,
                            const uint32_t *const *, const uint32_t *const *);

// SIMD backends consume packed architectural operands after public validation.
template <bool Bf16, int WaveSize, bool Packed> Swmmac16Backend swmmac16_x86_64_v3;
template <bool Bf8A, bool Bf8B, int WaveSize> Swmmac8Backend swmmac8_x86_64_v3;
template <bool Bf16, int WaveSize, bool Packed> Swmmac16Backend swmmac16_x86_64_v4;
template <bool Bf8A, bool Bf8B, int WaveSize> Swmmac8Backend swmmac8_x86_64_v4;

// Try the native FP8 pair-dot under eligible host controls and input accumulators.
// A decline leaves all destination registers untouched.
template <bool Bf8A, bool Bf8B, int Wave>
bool swmmac_fp8_x86_64_v4_bf16(uint32_t, uint32_t *const *, const uint32_t *const *,
                               const uint32_t *const *, const uint32_t *const *);

} // namespace rocjitsu::arith

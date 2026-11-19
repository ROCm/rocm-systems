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

// Accumulate decoded sparse inputs with loose arithmetic and write D in
// 32-lane sections; input.acc is also updated. Packed outputs narrow at each
// DOT4 stage; saturate controls finite FP16 overflow.
template <bool Bf16, bool Packed>
void swmmac_float_x86_64_v3(uint32_t mode, uint32_t *const *d, SwmmacFloatInputs &input,
                            bool saturate);

// Accumulate decoded sparse inputs with loose arithmetic and write D in
// 32-lane sections; input.acc is also updated. Packed outputs narrow at each
// DOT4 stage; saturate controls finite FP16 overflow.
template <bool Bf16, bool Packed>
void swmmac_float_x86_64_v4(uint32_t mode, uint32_t *const *d, SwmmacFloatInputs &input,
                            bool saturate);

} // namespace rocjitsu::arith

// SPDX-License-Identifier: MIT

// Declarations connecting dense FP16/BF16 WMMA dispatch to the bit-exact
// AVX2 and AVX-512 kernels. Covers both wave sizes, gfx11/gfx12 layouts, and
// FP32 or packed results without exposing intrinsics to the dispatch files.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Compute empirical bit-exact FP16/BF16 A*B+C for the selected gfx layout
// and wave size. Requires validated mode; D may alias inputs. saturate applies
// to finite FP16 overflow, and packed gfx11 preserves the unselected D half.
template <bool Bf16, int Lanes, bool Rdna3, bool Packed>
void wmma_exact_x86_64_v3(uint32_t mode, bool saturate, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c);

// Compute empirical bit-exact FP16/BF16 A*B+C for the selected gfx layout
// and wave size. Requires validated mode; D may alias inputs. saturate applies
// to finite FP16 overflow, and packed gfx11 preserves the unselected D half.
template <bool Bf16, int Lanes, bool Rdna3, bool Packed>
void wmma_exact_x86_64_v4(uint32_t mode, bool saturate, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c);

} // namespace rocjitsu::arith

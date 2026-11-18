// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 instantiation of the vectorised empirical bit-exact gfx12 FP8 WMMA;
// the algorithm lives in wmma_gfx12_fp8_exact.h.

#include "common/simd.h"
#include "wmma/x86_64/wmma_gfx12_fp8_exact.h"

#include <stdint.h>

namespace rocjitsu::arith {

// Compute empirical bit-exact FP8 A*B+C into FP32 D in gfx12 Wave32 layout.
// D may alias inputs.
void fp8_wmma_exact_x86_64_v3(uint32_t *const *d, const uint32_t *const *a,
                              const uint32_t *const *b, const uint32_t *const *c) {
  RJ_ARITH_IMPL_NAMESPACE::wmma_gfx12_fp8_exact::dense_avx2(d, a, b, c);
}

} // namespace rocjitsu::arith

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-isolated x86-64-v3 dense byte matrix entry points.
#include "mfma/backends.h"
#include "mfma/integer.h"
namespace rocjitsu::arith {
// Evaluate the full matrix using this translation unit's SIMD reduction.
template <int M, int K, int Batch, bool Wmma>
void matrix_i8_x86_64_v3(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                         const uint32_t *const *b, const uint32_t *const *c) {
  RJ_ARITH_IMPL_NAMESPACE::matrix_i8_kernel<M, K, Batch, Wmma>(controls, d, a, b, c);
}
template void matrix_i8_x86_64_v3<32, 4, 2, false>(uint64_t controls, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);
template void matrix_i8_x86_64_v3<16, 4, 4, false>(uint64_t controls, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);
template void matrix_i8_x86_64_v3<4, 4, 16, false>(uint64_t controls, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);
template void matrix_i8_x86_64_v3<32, 16, 1, false>(uint64_t controls, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
template void matrix_i8_x86_64_v3<16, 32, 1, false>(uint64_t controls, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
template void matrix_i8_x86_64_v3<16, 64, 1, false>(uint64_t controls, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
template void matrix_i8_x86_64_v3<32, 32, 1, false>(uint64_t controls, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
template void matrix_i8_x86_64_v3<16, 64, 1, true>(uint64_t controls, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);
} // namespace rocjitsu::arith

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-isolated x86-64-v4 dense byte matrix entry points.
#include "mfma/backends.h"
#include "mfma/integer.h"
namespace rocjitsu::arith {
// Evaluate the full matrix using this translation unit's SIMD reduction.
template <int M, int K, int Batch, bool Wmma, int ABlockLanes>
void matrix_i8_x86_64_v4(uint64_t controls, bool clamp_overflow, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c) {
  RJ_ARITH_IMPL_NAMESPACE::matrix_i8_kernel<M, K, Batch, Wmma, ABlockLanes>(
      controls, clamp_overflow, d, a, b, c);
}
template void matrix_i8_x86_64_v4<32, 4, 2, false>(uint64_t controls, bool clamp_overflow,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);
template void matrix_i8_x86_64_v4<16, 4, 4, false>(uint64_t controls, bool clamp_overflow,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);
template void matrix_i8_x86_64_v4<4, 4, 16, false>(uint64_t controls, bool clamp_overflow,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);
template void matrix_i8_x86_64_v4<32, 16, 1, false>(uint64_t controls, bool clamp_overflow,
                                                    uint32_t *const *d, const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
template void matrix_i8_x86_64_v4<16, 32, 1, false>(uint64_t controls, bool clamp_overflow,
                                                    uint32_t *const *d, const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
template void matrix_i8_x86_64_v4<16, 64, 1, false>(uint64_t controls, bool clamp_overflow,
                                                    uint32_t *const *d, const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
template void matrix_i8_x86_64_v4<32, 32, 1, false>(uint64_t controls, bool clamp_overflow,
                                                    uint32_t *const *d, const uint32_t *const *a,
                                                    const uint32_t *const *b,
                                                    const uint32_t *const *c);
template void matrix_i8_x86_64_v4<16, 64, 1, true>(uint64_t controls, bool clamp_overflow,
                                                   uint32_t *const *d, const uint32_t *const *a,
                                                   const uint32_t *const *b,
                                                   const uint32_t *const *c);

// Older full-wave shapes and gfx90a adjacent-block broadcasts share the reduction.
template void matrix_i8_x86_64_v4<32, 4, 2, false, 32>(uint64_t, bool, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void matrix_i8_x86_64_v4<16, 4, 4, false, 16>(uint64_t, bool, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void matrix_i8_x86_64_v4<4, 4, 16, false, 4>(uint64_t, bool, uint32_t *const *,
                                                      const uint32_t *const *,
                                                      const uint32_t *const *,
                                                      const uint32_t *const *);
template void matrix_i8_x86_64_v4<32, 8, 1, false, 0>(uint64_t, bool, uint32_t *const *,
                                                      const uint32_t *const *,
                                                      const uint32_t *const *,
                                                      const uint32_t *const *);
template void matrix_i8_x86_64_v4<32, 8, 1, false, 32>(uint64_t, bool, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void matrix_i8_x86_64_v4<16, 16, 1, false, 0>(uint64_t, bool, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *);
template void matrix_i8_x86_64_v4<16, 16, 1, false, 16>(uint64_t, bool, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
} // namespace rocjitsu::arith

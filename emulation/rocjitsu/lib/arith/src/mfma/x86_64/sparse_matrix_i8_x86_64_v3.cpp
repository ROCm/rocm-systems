// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86-64-v3 sparse byte matrices share the dense integer row reduction.
#include "mfma/backends.h"
#include "mfma/sparse_integer.h"
namespace rocjitsu::arith {
// Compute all tied accumulator outputs in this ISA domain.
template <int M, int K, bool Wmma>
void sparse_matrix_i8_x86_64_v3(uint64_t controls, bool clamp, uint32_t *const *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                const uint32_t *const *index) {
  RJ_ARITH_IMPL_NAMESPACE::sparse_matrix_i8_kernel<M, K, Wmma>(controls, clamp, d, a, b, index);
}
template void sparse_matrix_i8_x86_64_v3<16, 64, false>(uint64_t controls, bool clamp,
                                                        uint32_t *const *d,
                                                        const uint32_t *const *a,
                                                        const uint32_t *const *b,
                                                        const uint32_t *const *index);
template void sparse_matrix_i8_x86_64_v3<32, 32, false>(uint64_t controls, bool clamp,
                                                        uint32_t *const *d,
                                                        const uint32_t *const *a,
                                                        const uint32_t *const *b,
                                                        const uint32_t *const *index);
template void sparse_matrix_i8_x86_64_v3<16, 128, false>(uint64_t controls, bool clamp,
                                                         uint32_t *const *d,
                                                         const uint32_t *const *a,
                                                         const uint32_t *const *b,
                                                         const uint32_t *const *index);
template void sparse_matrix_i8_x86_64_v3<32, 64, false>(uint64_t controls, bool clamp,
                                                        uint32_t *const *d,
                                                        const uint32_t *const *a,
                                                        const uint32_t *const *b,
                                                        const uint32_t *const *index);
template void sparse_matrix_i8_x86_64_v3<16, 128, true>(uint64_t controls, bool clamp,
                                                        uint32_t *const *d,
                                                        const uint32_t *const *a,
                                                        const uint32_t *const *b,
                                                        const uint32_t *const *index);
} // namespace rocjitsu::arith

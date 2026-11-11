// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86-64-v4 FP64 MFMA, retaining the legacy column-wise fused reduction.
#include "mfma/backends.h"
#include "mfma/f64.h"
namespace rocjitsu::arith {
// Compute the full Wave64 result in this ISA domain.
template <int M, int Batch>
void mfma_f64_x86_64_v4(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *c) {
  RJ_ARITH_IMPL_NAMESPACE::mfma_f64_kernel<M, Batch>(controls, d, a, b, c);
}
template void mfma_f64_x86_64_v4<16, 1>(uint64_t controls, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c);
template void mfma_f64_x86_64_v4<4, 4>(uint64_t controls, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c);
} // namespace rocjitsu::arith

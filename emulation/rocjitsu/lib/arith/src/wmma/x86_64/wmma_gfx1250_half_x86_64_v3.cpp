// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 entry points for gfx1250 half-input WMMA.
#include "wmma/gfx1250_backends.h"
#include "wmma/x86_64/wmma_gfx1250_half.h"

namespace rocjitsu::arith {
// Evaluate a Wave32 matrix with this translation unit's ISA-isolated kernel.
template <Wmma1250Form Form>
void wmma_gfx1250_half_x86_64_v3(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                 const uint32_t *const *a, const uint32_t *const *b,
                                 const uint32_t *const *c) {
  RJ_ARITH_IMPL_NAMESPACE::wmma_gfx1250_half_kernel<Form>(flags, controls, d, a, b, c);
}
template void wmma_gfx1250_half_x86_64_v3<Wmma1250Form::F32F16>(uint64_t flags, uint64_t controls,
                                                                uint32_t *const *d,
                                                                const uint32_t *const *a,
                                                                const uint32_t *const *b,
                                                                const uint32_t *const *c);
template void wmma_gfx1250_half_x86_64_v3<Wmma1250Form::F32Bf16>(uint64_t flags, uint64_t controls,
                                                                 uint32_t *const *d,
                                                                 const uint32_t *const *a,
                                                                 const uint32_t *const *b,
                                                                 const uint32_t *const *c);
template void wmma_gfx1250_half_x86_64_v3<Wmma1250Form::F16>(uint64_t flags, uint64_t controls,
                                                             uint32_t *const *d,
                                                             const uint32_t *const *a,
                                                             const uint32_t *const *b,
                                                             const uint32_t *const *c);
template void wmma_gfx1250_half_x86_64_v3<Wmma1250Form::Bf16>(uint64_t flags, uint64_t controls,
                                                              uint32_t *const *d,
                                                              const uint32_t *const *a,
                                                              const uint32_t *const *b,
                                                              const uint32_t *const *c);
template void wmma_gfx1250_half_x86_64_v3<Wmma1250Form::Bf16F32>(uint64_t flags, uint64_t controls,
                                                                 uint32_t *const *d,
                                                                 const uint32_t *const *a,
                                                                 const uint32_t *const *b,
                                                                 const uint32_t *const *c);
} // namespace rocjitsu::arith

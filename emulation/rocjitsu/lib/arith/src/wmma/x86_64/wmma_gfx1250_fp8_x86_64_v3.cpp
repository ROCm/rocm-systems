// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// v3 entry points for gfx1250 OCP byte-input WMMA.
#include "wmma/gfx1250_backends.h"
#include "wmma/x86_64/wmma_gfx1250_fp8.h"
namespace rocjitsu::arith {
// Dispatch to this translation unit's ISA-isolated matrix kernel.
template <int K, bool Bf8A, bool Bf8B, bool Packed>
void wmma_gfx1250_fp8_x86_64_v3(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                const uint32_t *const *c) {
  RJ_ARITH_IMPL_NAMESPACE::wmma_gfx1250_fp8_kernel<K, Bf8A, Bf8B, Packed>(flags, controls, d, a, b,
                                                                          c);
}
template void wmma_gfx1250_fp8_x86_64_v3<64, false, false, false>(uint64_t flags, uint64_t controls,
                                                                  uint32_t *const *d,
                                                                  const uint32_t *const *a,
                                                                  const uint32_t *const *b,
                                                                  const uint32_t *const *c);
template void wmma_gfx1250_fp8_x86_64_v3<64, false, false, true>(uint64_t flags, uint64_t controls,
                                                                 uint32_t *const *d,
                                                                 const uint32_t *const *a,
                                                                 const uint32_t *const *b,
                                                                 const uint32_t *const *c);
template void wmma_gfx1250_fp8_x86_64_v3<64, false, true, false>(uint64_t flags, uint64_t controls,
                                                                 uint32_t *const *d,
                                                                 const uint32_t *const *a,
                                                                 const uint32_t *const *b,
                                                                 const uint32_t *const *c);
template void wmma_gfx1250_fp8_x86_64_v3<64, false, true, true>(uint64_t flags, uint64_t controls,
                                                                uint32_t *const *d,
                                                                const uint32_t *const *a,
                                                                const uint32_t *const *b,
                                                                const uint32_t *const *c);
template void wmma_gfx1250_fp8_x86_64_v3<64, true, false, false>(uint64_t flags, uint64_t controls,
                                                                 uint32_t *const *d,
                                                                 const uint32_t *const *a,
                                                                 const uint32_t *const *b,
                                                                 const uint32_t *const *c);
template void wmma_gfx1250_fp8_x86_64_v3<64, true, false, true>(uint64_t flags, uint64_t controls,
                                                                uint32_t *const *d,
                                                                const uint32_t *const *a,
                                                                const uint32_t *const *b,
                                                                const uint32_t *const *c);
template void wmma_gfx1250_fp8_x86_64_v3<64, true, true, false>(uint64_t flags, uint64_t controls,
                                                                uint32_t *const *d,
                                                                const uint32_t *const *a,
                                                                const uint32_t *const *b,
                                                                const uint32_t *const *c);
template void wmma_gfx1250_fp8_x86_64_v3<64, true, true, true>(uint64_t flags, uint64_t controls,
                                                               uint32_t *const *d,
                                                               const uint32_t *const *a,
                                                               const uint32_t *const *b,
                                                               const uint32_t *const *c);
template void wmma_gfx1250_fp8_x86_64_v3<128, false, false, false>(
    uint64_t flags, uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
    const uint32_t *const *b, const uint32_t *const *c);
template void wmma_gfx1250_fp8_x86_64_v3<128, false, false, true>(uint64_t flags, uint64_t controls,
                                                                  uint32_t *const *d,
                                                                  const uint32_t *const *a,
                                                                  const uint32_t *const *b,
                                                                  const uint32_t *const *c);
template void wmma_gfx1250_fp8_x86_64_v3<128, false, true, false>(uint64_t flags, uint64_t controls,
                                                                  uint32_t *const *d,
                                                                  const uint32_t *const *a,
                                                                  const uint32_t *const *b,
                                                                  const uint32_t *const *c);
template void wmma_gfx1250_fp8_x86_64_v3<128, false, true, true>(uint64_t flags, uint64_t controls,
                                                                 uint32_t *const *d,
                                                                 const uint32_t *const *a,
                                                                 const uint32_t *const *b,
                                                                 const uint32_t *const *c);
template void wmma_gfx1250_fp8_x86_64_v3<128, true, false, false>(uint64_t flags, uint64_t controls,
                                                                  uint32_t *const *d,
                                                                  const uint32_t *const *a,
                                                                  const uint32_t *const *b,
                                                                  const uint32_t *const *c);
template void wmma_gfx1250_fp8_x86_64_v3<128, true, false, true>(uint64_t flags, uint64_t controls,
                                                                 uint32_t *const *d,
                                                                 const uint32_t *const *a,
                                                                 const uint32_t *const *b,
                                                                 const uint32_t *const *c);
template void wmma_gfx1250_fp8_x86_64_v3<128, true, true, false>(uint64_t flags, uint64_t controls,
                                                                 uint32_t *const *d,
                                                                 const uint32_t *const *a,
                                                                 const uint32_t *const *b,
                                                                 const uint32_t *const *c);
template void wmma_gfx1250_fp8_x86_64_v3<128, true, true, true>(uint64_t flags, uint64_t controls,
                                                                uint32_t *const *d,
                                                                const uint32_t *const *a,
                                                                const uint32_t *const *b,
                                                                const uint32_t *const *c);
} // namespace rocjitsu::arith

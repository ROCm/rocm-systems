// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// x86-64-v4 packed U64 arithmetic, including source negation and integer saturation.
#include "integer/packed_u64.h"
#include "integer/packed_u64_impl.h"
namespace rocjitsu::arith {
// Execute validated Wave32 packed arithmetic, preserving inactive lanes and aliases.
template <PackedU64 Op>
void packed_u64_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c) {
  RJ_ARITH_IMPL_NAMESPACE::packed_u64_kernel<Op>(exec_mask, mode, d, a, b, c);
}
template void packed_u64_x86_64_v4<PackedU64::Add>(uint32_t, uint32_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void packed_u64_x86_64_v4<PackedU64::Sub>(uint32_t, uint32_t, uint32_t *const *,
                                                   const uint32_t *const *, const uint32_t *const *,
                                                   const uint32_t *const *);
template void packed_u64_x86_64_v4<PackedU64::ShiftAdd>(uint32_t, uint32_t, uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *,
                                                        const uint32_t *const *);
} // namespace rocjitsu::arith

// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1251 packed U64 API validation and scalar/AVX2/AVX-512 dispatch.
#include "integer/packed_u64.h"
#include "integer/packed_u64_impl.h"
#include "rj_arith/detail/rj_arith_integer.h"
namespace {
// Validate all instruction flags before accessing operands; all semantics share integer arithmetic.
template <rocjitsu::arith::PackedU64 Op>
int evaluate(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
             const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  constexpr uint32_t supported = Op == PackedU64::ShiftAdd
                                     ? 0
                                     : RJ_ARITH_PK_NEG_LO_A | RJ_ARITH_PK_NEG_LO_B |
                                           RJ_ARITH_PK_NEG_HI_A | RJ_ARITH_PK_NEG_HI_B |
                                           RJ_ARITH_PK_CLAMP;
  if (int error = validate(flags, mode & ~uint64_t(supported))) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    packed_u64_x86_64_v4<Op>(exec_mask, uint32_t(mode), d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    packed_u64_x86_64_v3<Op>(exec_mask, uint32_t(mode), d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
  packed_u64_kernel<Op>(exec_mask, uint32_t(mode), d, a, b, c);
  return RJ_ARITH_SUCCESS;
}
} // namespace
// Add packed U64 elements, applying validated independent sign and clamp controls.
int rj_arith_v_pk_add_nc_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b) {
  return evaluate<rocjitsu::arith::PackedU64::Add>(flags, exec_mask, mode, d, a, b, nullptr);
}
// Subtract packed U64 elements, applying validated independent sign and clamp controls.
int rj_arith_v_pk_sub_nc_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b) {
  return evaluate<rocjitsu::arith::PackedU64::Sub>(flags, exec_mask, mode, d, a, b, nullptr);
}
// Shift each packed U64 by its U32 count and add the corresponding packed addend.
int rj_arith_v_pk_lshl_add_u64_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, const uint32_t *const *c) {
  return evaluate<rocjitsu::arith::PackedU64::ShiftAdd>(flags, exec_mask, mode, d, a, b, c);
}

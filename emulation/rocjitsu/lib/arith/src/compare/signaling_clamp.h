// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared gfx908/gfx10 Compatibility policy: CLAMP signals on any participating
// NaN. Gfx10 constant predicates retain this rule in strict FP-status mode;
// numeric-only modes use the supplied comparator directly.
#pragma once

#include "common/isa.h"
#include "common/validation.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// The result-mask type selects Wave32 or Wave64 without changing VGPR layout.
template <typename Mask>
using NumericCompare = int (*)(uint64_t, Mask, uint64_t, Mask *, const uint32_t *const *,
                               const uint32_t *const *, uint32_t *);

// Compute the numeric and unordered masks before publishing outputs, allowing
// either output to overlap sources. Constant masks may report strict CLAMP status;
// other predicates retain their existing FP-status qualification. Constant is true
// only for F/T predicates, whose masks need no floating arithmetic.
template <bool Constant = false, typename Mask>
int compare_with_signaling_clamp(uint64_t flags, Mask exec_mask, uint64_t mode, Mask *d,
                                 const uint32_t *const *a, const uint32_t *const *b,
                                 uint32_t *trapsts, NumericCompare<Mask> numeric,
                                 NumericCompare<Mask> unordered) {
  const bool reporting = rocjitsu::arith::reports_fp_status(flags);
  const bool signal = ((flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY ||
                       (Constant && reporting)) &&
                      (mode & RJ_ARITH_ALU_CLAMP);
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (reporting) {
    if constexpr (!Constant) {
      return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
    }
    // Constant masks need no arithmetic qualification; CLAMP still classifies NaNs.
    flags =
        (flags & ~RJ_ARITH_SEMANTICS_MASK) | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;
  }
  // No classification or scratch result is needed on the ordinary numeric path.
  if (!signal) {
    return numeric(flags, exec_mask, mode, d, a, b, nullptr);
  }
  Mask result = 0, nans = 0;
  if (int error = numeric(flags, exec_mask, mode, &result, a, b, nullptr)) {
    return error;
  }
  // The unordered predicate supplies the same selected/DPP-filtered NaN mask
  // through the existing scalar/AVX2/AVX-512 arithmetic path.
  if (int error = unordered(flags, exec_mask, mode, &nans, a, b, nullptr)) {
    return error;
  }
  __builtin_memcpy(static_cast<void *>(d), &result, sizeof result);
  if (nans) {
    *trapsts |= RJ_ARITH_EXCEPTION_INVALID;
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE

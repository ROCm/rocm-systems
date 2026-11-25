// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Qualification boundary for the inherited gfx1150 scalar software model.
#pragma once
#include "common/validation.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Resolve Compatibility without claiming new gfx1150 empirical qualification.
// The caller retains Compatibility's status policy before resolving the flags.
inline int prepare_gfx1150_scalar(uint64_t &flags) {
  const uint64_t semantics = flags & RJ_ARITH_SEMANTICS_MASK;
  if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
      semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  return validate_semantics(flags);
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
